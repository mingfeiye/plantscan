"""
make_test_scan.py
Makes a simulated 3D scan, so that we can test plant_inventory.py without a phone.

Two scenes:
  easy  - a flat floor with 8 potted bushes (the Step 1 test).
  hard  - the known weak points of a real scan:
          * the plants stand on a TABLE, and the FLOOR under it is larger (more points),
          * a WALL behind the table and table legs,
          * two bushes that TOUCH each other,
          * one plant with only THIN LEAVES (few points),
          * optional: Y axis pointing DOWN, as in Scaniverse splat files.

Only the surface is sampled, like a real scan, with about 3 mm random error.
Units are meters. A fixed random seed gives the same scan every time.

Use (run in the project folder):
    python tools/make_test_scan.py test_scans/sim_scan.ply
    python tools/make_test_scan.py test_scans/sim_hard.ply --scene hard --y-down
It prints the true height and spread of each plant and writes them to <scan>.truth.json.
"""
import argparse
import json

import numpy as np
import trimesh

rng = np.random.default_rng(42)


def sphere_surface(n, center, radii):
    """n points on the surface of an ellipsoid."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return center + d * radii


def pot_surface(n, x, z, base_y, r_bottom, r_top, height):
    """n points on the side of a slightly tapered pot."""
    t = rng.uniform(0, 2 * np.pi, n)
    y = rng.uniform(0, height, n)
    r = r_bottom + (r_top - r_bottom) * y / height
    return np.column_stack([x + r * np.cos(t), base_y + y, z + r * np.sin(t)])


def rect(n, x0, x1, y0, y1, z0, z1):
    """n points uniformly in an axis-aligned box (use a zero range for a flat rectangle)."""
    return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)])


def bush(x, z, base_y, pot_h, pot_r, bush_r, top):
    """A pot with a round bush on top. Returns points, true height, true spread."""
    parts = [pot_surface(4000, x, z, base_y, pot_r * 0.8, pot_r, pot_h)]
    center_y = base_y + top - bush_r * 0.9
    b = sphere_surface(9000, np.array([x, center_y, z]), np.array([bush_r, bush_r * 0.9, bush_r]))
    parts.append(b[b[:, 1] > base_y + pot_h * 0.8])
    return np.vstack(parts), top, 2 * bush_r


def thin_leaf_plant(x, z, base_y, pot_h, pot_r, n_leaves, leaf_len):
    """A pot with thin strap leaves (like a grass or a young palm). Few points per leaf."""
    parts = [pot_surface(2500, x, z, base_y, pot_r * 0.8, pot_r, pot_h)]
    tips = []
    for k in range(n_leaves):
        ang = 2 * np.pi * k / n_leaves
        tilt = np.radians(25 + 10 * (k % 3))              # leaves lean outward 25..45 degrees
        direction = np.array([np.sin(tilt) * np.cos(ang), np.cos(tilt), np.sin(tilt) * np.sin(ang)])
        s = rng.uniform(0, leaf_len, 120)[:, None]          # only 120 points on each leaf
        start = np.array([x, base_y + pot_h, z])
        parts.append(start + s * direction)
        tips.append(start + leaf_len * direction)
    tips = np.array(tips)
    height = tips[:, 1].max() - base_y
    spread = 2 * np.max(np.hypot(tips[:, 0] - x, tips[:, 2] - z))
    return np.vstack(parts), height, spread


def easy_scene():
    plants = [(0.00, 0.00, 0.10, 0.07, 0.10, 0.22), (0.40, 0.05, 0.12, 0.08, 0.13, 0.31),
              (0.80, -0.02, 0.08, 0.06, 0.07, 0.14), (1.20, 0.03, 0.15, 0.10, 0.17, 0.45),
              (0.05, 0.45, 0.10, 0.07, 0.09, 0.19), (0.45, 0.50, 0.12, 0.08, 0.12, 0.27),
              (0.85, 0.42, 0.10, 0.07, 0.11, 0.24), (1.25, 0.48, 0.13, 0.09, 0.14, 0.36)]
    parts = [rect(120_000, -0.5, 1.7, 0, 0, -0.5, 1.1)]     # floor
    truth = []
    for (x, z, ph, pr, br, top) in plants:
        p, h, s = bush(x, z, 0.0, ph, pr, br, top)
        parts.append(p)
        truth.append({"x": x, "z": z, "height_cm": round(h * 100, 1), "spread_cm": round(s * 100, 1)})
    return parts, truth


def hard_scene():
    T = 0.75                                                # table top height
    parts = [
        rect(180_000, -1.0, 2.0, 0, 0, -1.2, 1.6),          # floor 3.0 x 2.8 m: the LARGEST plane
        rect(60_000, -0.2, 1.4, 0.0, 2.0, -0.70, -0.70),    # wall right behind the table
        rect(40_000, -0.15, 1.35, T, T, -0.65, 0.65),       # table top 1.5 x 1.3 m
    ]
    for lx in (-0.12, 1.32):                                # four table legs
        for lz in (-0.62, 0.62):
            parts.append(rect(1500, lx - 0.02, lx + 0.02, 0, T, lz - 0.02, lz + 0.02))
    truth = []
    # (x, z, pot height, pot radius, bush radius, total height)
    bushes = [(0.05, -0.30, 0.10, 0.07, 0.10, 0.22), (0.95, -0.35, 0.08, 0.06, 0.07, 0.14),
              (0.05, 0.30, 0.10, 0.07, 0.09, 0.19), (0.45, 0.35, 0.12, 0.08, 0.12, 0.27),
              (0.85, 0.30, 0.10, 0.07, 0.11, 0.24), (1.20, 0.30, 0.13, 0.09, 0.14, 0.36),
              # the TOUCHING pair: centres 0.25 m apart, bush radii 0.13 + 0.12 = 0.25 m
              (0.40, -0.30, 0.12, 0.08, 0.13, 0.31), (0.65, -0.30, 0.10, 0.07, 0.12, 0.25)]
    for (x, z, ph, pr, br, top) in bushes:
        p, h, s = bush(x, z, T, ph, pr, br, top)
        parts.append(p)
        truth.append({"x": x, "z": z, "height_cm": round(h * 100, 1), "spread_cm": round(s * 100, 1)})
    p, h, s = thin_leaf_plant(1.20, -0.30, T, 0.09, 0.07, 7, 0.28)
    parts.append(p)
    truth.append({"x": 1.20, "z": -0.30, "height_cm": round(h * 100, 1), "spread_cm": round(s * 100, 1),
                  "note": "thin leaves"})
    return parts, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="test_scans/sim_scan.ply")
    ap.add_argument("--scene", choices=["easy", "hard"], default="easy")
    ap.add_argument("--y-down", action="store_true", help="flip the Y axis, like a splat export")
    args = ap.parse_args()

    parts, truth = easy_scene() if args.scene == "easy" else hard_scene()
    pts = np.vstack(parts)
    pts += rng.normal(scale=0.003, size=pts.shape)          # about 3 mm scan noise
    if args.y_down:
        pts[:, 1] *= -1
    trimesh.PointCloud(pts).export(args.out)

    print(f"{'#':>2} {'true height (cm)':>17} {'true spread (cm)':>17}")
    for i, t in enumerate(truth, 1):
        print(f"{i:>2} {t['height_cm']:>17} {t['spread_cm']:>17}  {t.get('note', '')}")
    with open(args.out + ".truth.json", "w") as f:
        json.dump({"count": len(truth), "plants": truth}, f, indent=1)
    print(f"\nWrote {len(pts):,} points to {args.out} (and {args.out}.truth.json)")


if __name__ == "__main__":
    main()
