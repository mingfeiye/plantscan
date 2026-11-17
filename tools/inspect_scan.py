"""
inspect_scan.py
Prints what is inside a PLY scan file: the properties, the size of the scene on each axis,
and (for a Gaussian splat) the spread of blob size and opacity.

Use:
    python tools/inspect_scan.py scans/some_scan.ply
"""
import sys

import numpy as np

PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


def read_ply_vertices(path):
    """Read the vertex table of a binary little-endian PLY into a numpy structured array."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = [l for l in header if l.startswith("format")][0]
        if "binary_little_endian" not in fmt:
            raise ValueError(f"Unsupported PLY format: {fmt}")
        fields, n, in_vertex = [], 0, False
        for l in header:
            parts = l.split()
            if parts[:2] == ["element", "vertex"]:
                n, in_vertex = int(parts[2]), True
            elif parts and parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[0] == "property" and parts[1] != "list":
                fields.append((parts[2], PLY_TYPES[parts[1]]))
        return np.fromfile(f, dtype=np.dtype(fields), count=n)


def main():
    v = read_ply_vertices(sys.argv[1])
    names = v.dtype.names
    print(f"{len(v):,} vertices, {len(names)} properties")
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(float)
    for i, a in enumerate("xyz"):
        lo, p1, p50, p99, hi = np.percentile(xyz[:, i], [0, 1, 50, 99, 100])
        print(f"  {a}: min {lo:8.3f}  1% {p1:8.3f}  median {p50:8.3f}  99% {p99:8.3f}  max {hi:8.3f}")
    if "opacity" in names:
        op = 1 / (1 + np.exp(-v["opacity"].astype(float)))          # stored as logit
        sc = np.exp(np.column_stack([v[f"scale_{i}"] for i in range(3)]).astype(float))  # stored as log
        big = sc.max(axis=1)
        print("Gaussian splat:")
        print(f"  opacity  10%/50%/90%: {np.percentile(op, [10, 50, 90]).round(3)}")
        print(f"  largest blob axis 10%/50%/90%/99% (scene units): "
              f"{np.percentile(big, [10, 50, 90, 99]).round(4)}")
    if "nx" in names:
        nrm = np.column_stack([v["nx"], v["ny"], v["nz"]])
        print(f"  normals all zero: {bool(np.allclose(nrm, 0))}")


if __name__ == "__main__":
    main()
