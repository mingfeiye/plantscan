"""
make_test_scan.py
Makes a simulated 3D scan, so that we can test plant_inventory.py without a phone.

The fake scene: a flat floor, and 8 container plants (a pot with a round bush on top),
all with different heights. Only the surface is sampled, like a real scan, and a small
random error (about 3 mm) is added to each point. Units are meters and the Y axis is up,
like iPhone scan apps.

Use:
    python tools/make_test_scan.py test_scans/sim_scan.ply
It also prints the true height and spread of each plant, to compare with the result.
"""
import sys

import numpy as np
import trimesh

rng = np.random.default_rng(42)          # fixed seed: the same scan every time


def sphere_surface(n, center, radii):
    """n points on the surface of an ellipsoid."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return center + d * radii


def cylinder_surface(n, center_xz, r_bottom, r_top, height):
    """n points on the side and top rim of a slightly tapered pot."""
    t = rng.uniform(0, 2 * np.pi, n)
    y = rng.uniform(0, height, n)
    r = r_bottom + (r_top - r_bottom) * y / height
    return np.column_stack([center_xz[0] + r * np.cos(t), y, center_xz[1] + r * np.sin(t)])


# (x, z, pot height, pot radius, bush radius, bush top height)  -- meters
PLANTS = [
    (0.00, 0.00, 0.10, 0.07, 0.10, 0.22),
    (0.40, 0.05, 0.12, 0.08, 0.13, 0.31),
    (0.80, -0.02, 0.08, 0.06, 0.07, 0.14),
    (1.20, 0.03, 0.15, 0.10, 0.17, 0.45),
    (0.05, 0.45, 0.10, 0.07, 0.09, 0.19),
    (0.45, 0.50, 0.12, 0.08, 0.12, 0.27),
    (0.85, 0.42, 0.10, 0.07, 0.11, 0.24),
    (1.25, 0.48, 0.13, 0.09, 0.14, 0.36),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "test_scans/sim_scan.ply"
    parts = []

    # Floor: 2.2 m x 1.6 m around the plants.
    n_floor = 120_000
    parts.append(np.column_stack([rng.uniform(-0.5, 1.7, n_floor), np.zeros(n_floor),
                                  rng.uniform(-0.5, 1.1, n_floor)]))

    print(f"{'#':>2} {'true height (cm)':>17} {'true spread (cm)':>17}")
    for i, (x, z, pot_h, pot_r, bush_r, top) in enumerate(PLANTS, 1):
        parts.append(cylinder_surface(4000, (x, z), pot_r * 0.8, pot_r, pot_h))
        bush_center_y = top - bush_r * 0.9            # bush is a little flattened
        bush = sphere_surface(9000, np.array([x, bush_center_y, z]),
                              np.array([bush_r, bush_r * 0.9, bush_r]))
        bush = bush[bush[:, 1] > pot_h * 0.8]          # nothing inside the pot
        parts.append(bush)
        print(f"{i:>2} {top * 100:>17.1f} {2 * bush_r * 100:>17.1f}")

    pts = np.vstack(parts)
    pts += rng.normal(scale=0.003, size=pts.shape)    # about 3 mm scan noise
    trimesh.PointCloud(pts).export(out)
    print(f"\nWrote {len(pts):,} points to {out}")


if __name__ == "__main__":
    main()
