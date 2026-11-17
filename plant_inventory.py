"""
plant_inventory.py
Counts the objects (plants) in a 3D scan and measures the height and width of each one.

Use:
    python3 plant_inventory.py scan.ply
Output:
    - a table in the terminal
    - inventory.csv  (one row for each plant)
    - inventory.png  (top view, each plant with its number and height)

Units: the script assumes that the scan is in meters (iPhone scan apps export meters).
"""
import argparse
import csv
import sys

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_points(path):
    """Read the vertices of a point cloud or a mesh (PLY, OBJ, GLB, STL)."""
    data = trimesh.load(path)
    if isinstance(data, trimesh.Scene):
        parts = [np.asarray(g.vertices) for g in data.dump()]
        pts = np.vstack(parts)
    else:
        pts = np.asarray(data.vertices)
    return pts.astype(float)


def voxel_downsample(pts, size):
    """Keep one point for each cube of side `size` (meters)."""
    keys = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def fit_plane_ransac(pts, threshold, iterations=400, rng=None):
    """Return (normal, d, inlier_mask) of the plane with the most points."""
    rng = rng or np.random.default_rng(0)
    best_mask, best_n, best_d = None, None, None
    best_count = -1
    n_pts = len(pts)
    for _ in range(iterations):
        sample = pts[rng.choice(n_pts, 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        d = -normal.dot(sample[0])
        mask = np.abs(pts @ normal + d) < threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask, best_n, best_d = count, mask, normal, d
    return best_n, best_d, best_mask


def find_ground(pts, threshold):
    """Find the support surface (floor or table).
    The script prefers a plane that is almost horizontal (iPhone scans have the Y axis up)."""
    remaining = pts
    candidates = []
    for _ in range(3):
        if len(remaining) < 500:
            break
        normal, d, mask = fit_plane_ransac(remaining, threshold)
        candidates.append((normal, d, int(mask.sum())))
        remaining = remaining[~mask]
    horizontal = [c for c in candidates if abs(c[0][1]) > 0.85]
    normal, d, _ = max(horizontal or candidates, key=lambda c: c[2])
    # Make the normal point to the side where the objects are.
    heights = pts @ normal + d
    if np.sum(heights > threshold) < np.sum(heights < -threshold):
        normal, d = -normal, -d
    return normal, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="scan file: .ply, .obj, .glb or .stl")
    ap.add_argument("--min-height", type=float, default=0.04,
                    help="ignore objects lower than this (meters). Default 0.04")
    ap.add_argument("--max-width", type=float, default=1.0,
                    help="ignore objects wider than this (meters), for example walls. Default 1.0")
    ap.add_argument("--gap", type=float, default=0.03,
                    help="two points nearer than this are the same object (meters). Default 0.03")
    ap.add_argument("--grades", type=float, nargs="+", default=[0.15, 0.30],
                    help="height limits between grades (meters). Default 0.15 0.30")
    args = ap.parse_args()

    pts = load_points(args.scan)
    print(f"Loaded {len(pts):,} points from {args.scan}")
    pts = voxel_downsample(pts, 0.01)
    print(f"After downsample to 1 cm: {len(pts):,} points")

    normal, d = find_ground(pts, threshold=0.015)
    height = pts @ normal + d

    # Axes in the ground plane, for the top view.
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    above = height > 0.02
    obj_pts, obj_h = pts[above], height[above]
    if len(obj_pts) == 0:
        sys.exit("No points above the ground plane. Check the scan.")

    labels = DBSCAN(eps=args.gap, min_samples=8).fit_predict(obj_pts)

    plants = []
    for lab in sorted(set(labels) - {-1}):
        m = labels == lab
        p, h = obj_pts[m], obj_h[m]
        top = float(np.percentile(h, 99))          # 99th percentile, to ignore stray points
        if h.min() > 0.06:                         # the object does not touch the ground
            continue
        pu, pv = p @ u, p @ v
        width = float(max(np.ptp(pu), np.ptp(pv)))
        if top < args.min_height or width > args.max_width:
            continue
        plants.append({"x": float(pu.mean()), "y": float(pv.mean()),
                       "height_cm": round(top * 100, 1), "width_cm": round(width * 100, 1)})

    # Number the plants from left to right.
    plants.sort(key=lambda r: r["x"])
    limits = sorted(args.grades)
    names = ["small", "medium", "large", "extra large", "grade 5", "grade 6"]
    for i, r in enumerate(plants, 1):
        r["id"] = i
        r["grade"] = names[int(np.searchsorted(limits, r["height_cm"] / 100))]

    print(f"\nPlants found: {len(plants)}\n")
    print(f"{'ID':>3} {'Height (cm)':>12} {'Width (cm)':>11}  Grade")
    for r in plants:
        print(f"{r['id']:>3} {r['height_cm']:>12} {r['width_cm']:>11}  {r['grade']}")
    print("\nCount for each grade:")
    for g in names:
        c = sum(1 for r in plants if r["grade"] == g)
        if c:
            print(f"  {g}: {c}")

    with open("inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "height_cm", "width_cm", "grade"], extrasaction="ignore")
        w.writeheader(); w.writerows(plants)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(obj_pts @ u, obj_pts @ v, c=obj_h * 100, s=2, cmap="viridis")
    for r in plants:
        ax.annotate(f"#{r['id']}\n{r['height_cm']} cm", (r["x"], r["y"]), ha="center",
                    fontsize=9, color="white", bbox=dict(boxstyle="round", fc="black", alpha=0.6))
    ax.set_aspect("equal"); ax.set_title(f"Top view: {len(plants)} plants")
    ax.set_xlabel("meters"); ax.set_ylabel("meters")
    fig.colorbar(sc, label="height above ground (cm)")
    fig.savefig("inventory.png", dpi=150, bbox_inches="tight")
    print("\nWrote inventory.csv and inventory.png")


if __name__ == "__main__":
    main()
