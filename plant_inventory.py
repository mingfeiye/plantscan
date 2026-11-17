"""
plant_inventory.py
Counts the objects (plants) in a 3D scan and measures the height and spread of each one.

Use (run in the project folder):
    venv/bin/python plant_inventory.py scans/my_scan.ply

Input:  a Gaussian splat PLY (Scaniverse splat export, the main input),
        or a mesh / point cloud (PLY, OBJ, GLB).
Output (in the folder output/):
    inventory.csv   one row for each plant
    inventory.json  the same data plus scan details, for the next steps
    inventory.png   top view, each plant with its number
If the scan cannot be measured reliably, the program prints "SCAN PROBLEM: ..." and
writes NO results (old results are deleted), so that no wrong numbers are shown.

Units: meters inside the program, centimeters in the results.

How it works:
 1. Load the points. For a splat: keep the blob centres, remove faint blobs and the
    huge background blobs (the "sky shell").
 2. Find the flat horizontal surfaces (RANSAC) and decide which way is UP
    (Scaniverse splats have Y down, meshes have Y up; the program checks, it does not assume).
 3. Try each surface (table, floor, ...) and keep the one that the objects stand on.
 4. Group the points above the surface into objects (DBSCAN), reject walls and furniture,
    and split two plants that touch (two height peaks with a valley between them).
 5. Measure each object: height = top above the surface; spread = average width over all
    directions (for a round plant this is the diameter).
"""
import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings (meters)
VOXEL = 0.005            # keep one point for each 5 mm cube
PLANE_TOL = 0.01         # a point within 1 cm of a plane belongs to it
ABOVE = 0.012            # points lower than 1.2 cm above the surface are the surface
NEIGHBOR_R = 0.02        # outlier filter: a point needs some neighbours within 2 cm ...
NEIGHBOR_MIN = 5         # ... at least this many (itself included)
MAX_OBJECT_HEIGHT = 1.5  # ignore everything higher than this above the surface
TOUCH = 0.06             # an object must reach down to within 6 cm of the surface
SPLIT_MIN_SEP = 0.08     # two plant tops must be at least 8 cm apart to split
SPLIT_VALLEY = 0.65      # ... and the valley between them lower than 65% of the lower top
GRID = 0.01              # 1 cm cells for the top-view height map
FOOTPRINT_CELL = 0.02    # 2 cm cells for the outline of the surface


class ScanError(Exception):
    """The scan cannot be measured reliably. The message says why, in plain words."""


# ---------------------------------------------------------------- loading
PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_ply_header(path):
    with open(path, "rb") as f:
        lines = []
        while True:
            raw = f.readline()
            if not raw:
                raise ScanError(f"{path} is not a complete PLY file (no end_header).")
            line = raw.decode("ascii", "replace").strip()
            lines.append(line)
            if line == "end_header":
                return lines, f.tell()


def is_splat(path):
    if Path(path).suffix.lower() != ".ply":
        return False
    header, _ = read_ply_header(path)
    props = {l.split()[-1] for l in header if l.startswith("property")}
    return {"opacity", "scale_0", "f_dc_0"} <= props


def read_ply_vertices(path):
    """The vertex table of a binary little-endian PLY, as a numpy structured array."""
    header, offset = read_ply_header(path)
    fmt = next(l for l in header if l.startswith("format"))
    if "binary_little_endian" not in fmt:
        raise ScanError(f"Unsupported PLY format for a splat: {fmt}")
    fields, n, in_vertex = [], 0, False
    for l in header:
        parts = l.split()
        if parts[:2] == ["element", "vertex"]:
            n, in_vertex = int(parts[2]), True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            if parts[1] == "list":
                raise ScanError("Unsupported PLY: list property in the vertex table.")
            fields.append((parts[2], PLY_TYPES[parts[1]]))
    with open(path, "rb") as f:
        f.seek(offset)
        return np.fromfile(f, dtype=np.dtype(fields), count=n)


def load_splat(path, min_opacity=0.3, max_blob=0.05):
    """Blob centres of a Gaussian splat. Faint blobs (opacity < 0.3) and huge background
    blobs (larger than 5 cm) are removed."""
    v = read_ply_vertices(path)
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(float)
    opacity = 1 / (1 + np.exp(-v["opacity"].astype(float)))          # stored as logit
    size = np.exp(np.column_stack([v[f"scale_{i}"] for i in range(3)]).astype(float)).max(1)
    background = size > max_blob
    keep = (opacity >= min_opacity) & ~background & np.isfinite(xyz).all(1)
    info = {"kind": "Gaussian splat", "elements": int(len(v)),
            "background_blobs_removed": int(background.sum()),
            "faint_blobs_removed": int(((opacity < min_opacity) & ~background).sum())}
    return xyz[keep], info


def load_mesh_or_cloud(path, density=40_000, max_samples=3_000_000):
    """Points of a mesh (vertices plus points sampled evenly on the surface, so that thin
    leaves with few vertices still get points) or of a point cloud."""
    data = trimesh.load(path)
    geoms = data.dump() if isinstance(data, trimesh.Scene) else [data]
    parts, kind = [], "point cloud"
    for g in geoms:
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            kind = "mesh"
            count = int(min(max_samples, max(1000, g.area * density)))
            sampled, _ = trimesh.sample.sample_surface(g, count, seed=0)
            parts += [np.asarray(g.vertices), np.asarray(sampled)]
        elif hasattr(g, "vertices"):
            parts.append(np.asarray(g.vertices))
    if not parts:
        raise ScanError("The file contains no 3D points.")
    return np.vstack(parts).astype(float), {"kind": kind, "elements": int(sum(len(p) for p in parts))}


def load_points(path):
    if not Path(path).exists():
        raise ScanError(f"File not found: {path}")
    return load_splat(path) if is_splat(path) else load_mesh_or_cloud(path)


def voxel_downsample(pts, size):
    keys = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


# ---------------------------------------------------------------- planes
def fit_plane_ransac(pts, threshold, iterations=500, seed=0):
    """(normal, d, inlier mask) of the plane with the most points: normal . x + d = 0."""
    rng = np.random.default_rng(seed)
    best = (None, None, None, -1)
    for _ in range(iterations):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        d = -n.dot(s[0])
        mask = np.abs(pts @ n + d) < threshold
        c = int(mask.sum())
        if c > best[3]:
            best = (n, d, mask, c)
    n, d, mask, _ = best
    # Refine with a least-squares fit on the inliers.
    q = pts[mask]
    centre = q.mean(0)
    n2 = np.linalg.svd(q - centre, full_matrices=False)[2][-1]
    if n2.dot(n) < 0:
        n2 = -n2
    d2 = -n2.dot(centre)
    return n2, d2, np.abs(pts @ n2 + d2) < threshold


def horizontal_planes(pts, max_planes=5):
    """The large planes that are parallel to the largest near-horizontal one."""
    planes, remaining = [], pts
    for k in range(max_planes):
        if len(remaining) < 2000:
            break
        n, d, mask = fit_plane_ransac(remaining, PLANE_TOL, seed=k)
        planes.append((n, d, int(mask.sum())))
        remaining = remaining[~mask]
    # Phone scans keep the vertical axis on the file's Y axis (up or down).
    flat = [p for p in planes if abs(p[0][1]) > 0.85]
    if not flat:
        raise ScanError("No flat horizontal surface (table or floor) was found in the scan.")
    ref = max(flat, key=lambda p: p[2])[0]
    out = []
    for n, d, c in flat:
        if abs(n.dot(ref)) > np.cos(np.radians(10)):         # parallel within 10 degrees
            if n.dot(ref) < 0:
                n, d = -n, -d
            out.append((n, d, c))
    return out


def decide_up(pts, planes):
    """Up is the side where most points are, judged at the most one-sided plane
    (the lowest surface has almost nothing under it). Returns +1 or -1 and how sure."""
    best = None
    for n, d, _ in planes:
        h = pts @ n + d
        above, below = int((h > 0.03).sum()), int((h < -0.03).sum())
        ratio = (max(above, below) + 1) / (min(above, below) + 1)
        if best is None or ratio > best[0]:
            best = (ratio, 1 if above >= below else -1)
    return best[1], best[0]


# ---------------------------------------------------------------- top view helpers
def top_axes(normal):
    """Two horizontal axes for the top view: u follows the file's X axis, w = normal x u."""
    x = np.array([1.0, 0, 0])
    u = x - normal * normal.dot(x)
    if np.linalg.norm(u) < 0.1:
        z = np.array([0, 0, 1.0])
        u = z - normal * normal.dot(z)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def footprint(su, sw):
    """Outline of the support surface in the top view (holes under objects are filled)."""
    origin = np.array([su.min(), sw.min()]) - 0.2
    iu = ((su - origin[0]) / FOOTPRINT_CELL).astype(int)
    iw = ((sw - origin[1]) / FOOTPRINT_CELL).astype(int)
    grid = np.zeros((iu.max() + 21, iw.max() + 21), bool)
    grid[iu, iw] = True
    grid = ndimage.binary_closing(grid, iterations=4)
    grid = ndimage.binary_fill_holes(grid)
    grid = ndimage.binary_opening(grid, iterations=2)          # drop thin stray strips
    grid = ndimage.binary_dilation(grid, iterations=2)         # allow 4 cm of overhang

    def inside(pu, pw):
        a = ((pu - origin[0]) / FOOTPRINT_CELL).astype(int)
        b = ((pw - origin[1]) / FOOTPRINT_CELL).astype(int)
        ok = (a >= 0) & (b >= 0) & (a < grid.shape[0]) & (b < grid.shape[1])
        res = np.zeros(len(pu), bool)
        res[ok] = grid[a[ok], b[ok]]
        return res

    return inside, float(grid.sum() * FOOTPRINT_CELL ** 2)


def widths(pu, pw):
    """Widths of the object seen from above, in 90 directions: (mean, max, min).
    The mean width is the 'spread': for a round plant it equals the diameter.
    Each width uses the 0.5% and 99.5% positions, not the extremes, so that a few noisy
    points at the edge do not make the plant look wider."""
    pts2 = np.column_stack([pu, pw])
    ang = np.linspace(0, np.pi, 90, endpoint=False)
    proj = pts2 @ np.vstack([np.cos(ang), np.sin(ang)])
    lo, hi = np.percentile(proj, [0.5, 99.5], axis=0)
    w = hi - lo
    return float(w.mean()), float(w.max()), float(w.min())


# ---------------------------------------------------------------- splitting touching plants
def split_touching(pu, pw, h):
    """Split one cluster into several plants when its top-view shape is two or more round
    parts joined by a narrow neck (like the figure 8). Returns (labels, n_parts).

    Method: fill the top-view outline on a 1 cm grid and compute, for each cell, the
    distance to the edge of the outline. The centre of each round part is a peak of that
    distance; its value is the radius of the part. Two peaks are two plants when they are
    at least 8 cm apart, each part is at least 4 cm in radius, and the neck between them is
    narrower than 65% of the smaller radius. Thin leaves are narrow, so they never form a
    separate part."""
    o = np.array([pu.min(), pw.min()]) - 3 * GRID
    iu = ((pu - o[0]) / GRID).astype(int)
    iw = ((pw - o[1]) / GRID).astype(int)
    occ = np.zeros((iu.max() + 4, iw.max() + 4), bool)
    occ[iu, iw] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=2))
    dist = ndimage.distance_transform_edt(occ)                      # in cells
    dist_s = ndimage.gaussian_filter(dist, 1.0)
    sep = int(round(SPLIT_MIN_SEP / GRID))
    min_r = 0.04 / GRID
    is_peak = (dist_s == ndimage.maximum_filter(dist_s, size=sep + 1)) & (dist_s >= min_r)
    cand = sorted(zip(*np.nonzero(is_peak)), key=lambda c: -dist_s[c])

    def neck(a, b):
        t = np.linspace(0, 1, 80)
        line = np.outer(1 - t, a) + np.outer(t, b)
        return dist[line[:, 0].round().astype(int), line[:, 1].round().astype(int)].min()

    peaks = []
    for c in cand:
        c = np.array(c, float)
        r_c = dist_s[tuple(c.astype(int))]
        if all(np.linalg.norm(c - p) >= sep and neck(p, c) < SPLIT_VALLEY * min(r_c, r_p)
               for p, r_p in peaks):
            peaks.append((c, r_c))
    if len(peaks) < 2:
        return np.zeros(len(pu), int), 1
    # A point goes to the part with the smallest (distance to centre / radius),
    # so a larger plant gets a larger share.
    cells = np.column_stack([iu, iw]).astype(float)
    score = np.column_stack([np.linalg.norm(cells - c, axis=1) / r for c, r in peaks])
    return score.argmin(1), len(peaks)


def attach_floating(labels, U, W, H):
    """Thin leaves or foliage can be separated from their pot by a small gap, so they form a
    group that 'floats'. Such a group joins the standing group under it when its lowest
    points are within 5 cm (seen from above) of that group and at most 8 cm above its top.
    Small groups (fewer than 50 points) are stray splat blobs ("floaters"), not leaves.
    Every test uses the ORIGINAL groups, so that floaters cannot form a chain upward."""
    original = labels
    groups = sorted(set(labels) - {-1})
    standing = [g for g in groups if H[labels == g].min() <= TOUCH]
    if not standing:
        return labels
    labels = labels.copy()
    for g in groups:
        m = original == g
        if H[m].min() <= TOUCH or m.sum() < 50:
            continue
        low = m & (H < H[m].min() + 0.03)                         # bottom 3 cm of the group
        best = None
        for s in standing:
            ms = original == s
            dist, _ = cKDTree(np.column_stack([U[ms], W[ms]])).query(np.column_stack([U[low], W[low]]))
            gap_up = H[m].min() - H[ms].max()
            if dist.min() < 0.05 and gap_up < 0.08 and (best is None or dist.min() < best[0]):
                best = (dist.min(), s)
        if best:
            labels[m] = best[1]
    return labels


# ---------------------------------------------------------------- detection on one surface
def detect_on_surface(pts, normal, d, gap):
    h = pts @ normal + d
    u, w = top_axes(normal)
    pu, pw = pts @ u, pts @ w
    support = np.abs(h) < PLANE_TOL
    inside, area = footprint(pu[support], pw[support])

    # A real table or floor hides what is under it. A "plane" that is only a horizontal
    # slice through plants has many points just below it: reject it.
    below = int(((h < -ABOVE) & (h > -0.15) & inside(pu, pw)).sum())
    if below > 0.5 * support.sum():
        return {"objects": [], "rejected": [], "area_m2": area, "not_a_surface": True}

    # Points above the surface, over (or near) the surface outline.
    sel = (h > ABOVE) & (h < MAX_OBJECT_HEIGHT) & inside(pu, pw)
    if sel.sum() < 50:
        return {"objects": [], "rejected": [], "area_m2": area}
    P, Hh, U, W = pts[sel], h[sel], pu[sel], pw[sel]

    # Remove floating stray points (few neighbours).
    counts = np.array([len(x) for x in cKDTree(P).query_ball_point(P, NEIGHBOR_R)])
    good = counts >= NEIGHBOR_MIN
    P, Hh, U, W = P[good], Hh[good], U[good], W[good]
    if len(P) < 50:
        return {"objects": [], "rejected": [], "area_m2": area}

    labels = DBSCAN(eps=gap, min_samples=10).fit_predict(P)
    labels = attach_floating(labels, U, W, Hh)
    objects, rejected = [], []
    for lab in sorted(set(labels) - {-1}):
        m = labels == lab
        cu, cw, ch = U[m], W[m], Hh[m]
        if ch.min() > TOUCH:
            rejected.append("does not stand on the surface")
            continue
        parts, n_parts = split_touching(cu, cw, ch)
        for k in range(n_parts):
            mk = parts == k
            if mk.sum() < 30:
                continue
            ku, kw, kh = cu[mk], cw[mk], ch[mk]
            top = float(np.percentile(kh, 99.5))
            spread, wmax, wmin = widths(ku, kw)
            reason = None
            if kh.min() > TOUCH:
                reason = "does not stand on the surface"
            elif wmax > 0.25 and wmin < 0.15 * wmax:
                reason = "thin and long: probably a wall, a board or an edge"
            elif top > 1.2 or wmax > 1.2:
                reason = "too large: probably furniture"
            if reason:
                rejected.append(reason)
                continue
            objects.append({"u": float(ku.mean()), "w": float(kw.mean()),
                            "height_cm": round(top * 100, 1), "spread_cm": round(spread * 100, 1),
                            "max_width_cm": round(wmax * 100, 1), "points": int(mk.sum()),
                            "split_from_touching": n_parts > 1,
                            "outline": np.column_stack([ku, kw])})
    return {"objects": objects, "rejected": rejected, "area_m2": area,
            "view": (U, W, Hh), "support_uv": (pu[support], pw[support])}


# ---------------------------------------------------------------- grades
def load_grades(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    grades = cfg.get("grades", [])
    if not grades:
        raise ScanError(f"No grades in {path}.")
    return grades, bool(cfg.get("demo_values", False))


def grade_of(height_cm, grades):
    for g in grades:
        if "max_height_cm" not in g or height_cm <= g["max_height_cm"]:
            return g["name"]
    return grades[-1]["name"]


# ---------------------------------------------------------------- whole run
def analyse(scan, grades_path="config/grades.toml", gap=0.025, min_height=0.04, log=print):
    pts, info = load_points(scan)
    log(f"Loaded {scan}: {info['kind']}, {info['elements']:,} elements")
    if "background_blobs_removed" in info:
        log(f"  removed {info['background_blobs_removed']:,} background blobs and "
            f"{info['faint_blobs_removed']:,} faint blobs")
    pts = voxel_downsample(pts, VOXEL)
    log(f"  {len(pts):,} points after keeping one point per 5 mm")
    if len(pts) < 2000:
        raise ScanError(f"Only {len(pts)} usable points. The scan is too small or empty.")

    planes = horizontal_planes(pts)
    sign, sure = decide_up(pts, planes)
    planes = [(n * sign, d * sign, c) for n, d, c in planes]
    axis = "XYZ"[int(np.argmax(np.abs(planes[0][0])))]
    up_text = f"{'+' if planes[0][0][np.argmax(np.abs(planes[0][0]))] > 0 else '-'}{axis}"
    log(f"  up direction: {up_text} of the file (confidence ratio {sure:.1f})")

    # Try every horizontal surface; keep the one with the most objects standing on it.
    best = None
    for n, d, c in planes:
        res = detect_on_surface(pts, n, d, gap)
        res["objects"] = [o for o in res["objects"] if o["height_cm"] >= min_height * 100]
        level = -d                               # height of this surface along the up axis
        key = (len(res["objects"]), level)
        if res.get("not_a_surface"):
            log(f"  plane at {level:+.2f} m ({c:,} points): not a real surface (points under it)")
            continue
        log(f"  surface at {level:+.2f} m ({c:,} points): {len(res['objects'])} objects")
        if best is None or key > best[0]:
            best = (key, res, n, d)
    if best is None:
        raise ScanError("No flat surface was found that objects could stand on.")
    _, res, normal, d = best
    if not res["objects"]:
        raise ScanError("No objects were found standing on a flat surface.")
    if res["area_m2"] < 0.05:
        raise ScanError("The flat surface under the objects is too small to trust "
                        f"({res['area_m2']:.2f} m2).")

    grades, demo = load_grades(grades_path)
    objs = sorted(res["objects"], key=lambda o: (o["u"], o["w"]))   # left to right in the image
    for i, o in enumerate(objs, 1):
        o["id"] = i
        o["grade"] = grade_of(o["height_cm"], grades)
    counts = {g["name"]: sum(o["grade"] == g["name"] for o in objs) for g in grades}
    return {"scan": str(scan), "scan_info": info, "up": up_text, "up_confidence": round(sure, 1),
            "surface_area_m2": round(res["area_m2"], 2), "objects": objs,
            "rejected": res["rejected"], "grade_counts": counts, "grades_are_demo_values": demo,
            "_view": res["view"], "_support": res["support_uv"]}


def write_outputs(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    fields = ["id", "height_cm", "spread_cm", "max_width_cm", "grade", "points", "split_from_touching"]
    with open(out_dir / "inventory.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(result["objects"])
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    clean["objects"] = [{k: v for k, v in o.items() if k != "outline"} for o in result["objects"]]
    with open(out_dir / "inventory.json", "w") as f:
        json.dump(clean, f, indent=1)

    U, W, H = result["_view"]
    su, sw = result["_support"]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(su, sw, s=1, c="#d9d4c7")
    sc = ax.scatter(U, W, c=np.clip(H * 100, 0, None), s=2, cmap="viridis")
    for o in result["objects"]:
        ol = o["outline"]
        try:
            hull = ol[ConvexHull(ol).vertices]
            ax.fill(hull[:, 0], hull[:, 1], fill=False, ec="red", lw=1.5)
        except Exception:
            pass
        ax.annotate(str(o["id"]), (o["u"], o["w"]), ha="center", va="center", fontsize=20,
                    weight="bold", color="white",
                    bbox=dict(boxstyle="circle", fc="black", ec="white", alpha=0.8))
    pad = 0.1
    allu = np.concatenate([o["outline"][:, 0] for o in result["objects"]])
    allw = np.concatenate([o["outline"][:, 1] for o in result["objects"]])
    ax.set_xlim(allu.min() - pad, allu.max() + pad)
    ax.set_ylim(allw.min() - pad, allw.max() + pad)
    ax.set_aspect("equal")
    ax.set_title(f"Top view: {len(result['objects'])} plants", fontsize=18)
    ax.set_xlabel("meters"); ax.set_ylabel("meters")
    fig.colorbar(sc, label="height above surface (cm)")
    fig.savefig(out_dir / "inventory.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("scan", help="scan file: splat .ply, or mesh/point cloud .ply .obj .glb")
    ap.add_argument("--out", default="output", help="folder for the results. Default: output")
    ap.add_argument("--grades", default="config/grades.toml", help="grade limits file")
    ap.add_argument("--gap", type=float, default=0.025,
                    help="points nearer than this are the same object (meters). Default 0.025")
    ap.add_argument("--min-height", type=float, default=0.04,
                    help="ignore objects lower than this (meters). Default 0.04")
    args = ap.parse_args()

    out = Path(args.out)
    for name in ("inventory.csv", "inventory.json", "inventory.png"):   # never show old results
        (out / name).unlink(missing_ok=True)
    try:
        result = analyse(args.scan, args.grades, args.gap, args.min_height)
    except ScanError as e:
        print(f"\nSCAN PROBLEM: {e}\nNo results were written. Please scan again.")
        sys.exit(2)
    except Exception as e:                     # any other failure: still no wrong numbers
        print(f"\nSCAN PROBLEM: the scan could not be processed ({type(e).__name__}: {e}).\n"
              "No results were written. Please scan again.")
        sys.exit(2)

    objs = result["objects"]
    print(f"\nPlants found: {len(objs)}   (surface area {result['surface_area_m2']} m2)\n")
    print(f"{'ID':>3} {'Height cm':>10} {'Spread cm':>10} {'Max width':>10}  Grade")
    for o in objs:
        flag = "  (split from touching group)" if o["split_from_touching"] else ""
        print(f"{o['id']:>3} {o['height_cm']:>10} {o['spread_cm']:>10} {o['max_width_cm']:>10}  {o['grade']}{flag}")
    print("\nCount for each grade" + (" (DEMO grade limits)" if result["grades_are_demo_values"] else "") + ":")
    for g, c in result["grade_counts"].items():
        print(f"  {g}: {c}")
    write_outputs(result, out)
    print(f"\nWrote {out}/inventory.csv, inventory.json, inventory.png")


if __name__ == "__main__":
    main()
