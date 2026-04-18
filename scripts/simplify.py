#!/usr/bin/env python3
"""
Simplify a .stl or .3mf mesh by vertex clustering.

This is NOT quadric edge collapse (which requires per-vertex error metrics and
priority queues — ~500 lines and hard to keep pure-numpy). It's vertex-cluster
simplification: bin vertices to a grid, snap each to its cluster centroid,
drop degenerate triangles. That's the same algorithm used by Blender's
"Decimate → Un-subdivide" approximation and by many quick-preview tools.

Quality is good enough for print previews, LODs, and shipping smaller files
online. For production-quality decimation use Blender or MeshLab.

Usage:
    simplify.py <in> <out> --ratio 0.5              # keep ~50% of triangles
    simplify.py <in> <out> --max-tris 20000         # target <= 20k tris
    simplify.py <in> <out> --grid 0.5               # cluster at 0.5mm grid
"""
import sys, os, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _3mf_reader import read_stl, read_3mf, write_stl_binary, load_meshes


def cluster_simplify(tris_nx3x3, grid_size):
    """Snap verts to `grid_size`-mm cubes, drop degenerate triangles.

    Returns (new_tris, stats_dict)."""
    flat = tris_nx3x3.reshape(-1, 3).astype(np.float64)
    snapped = np.round(flat / grid_size) * grid_size

    # Re-assemble as triangles of snapped vertices
    new_tris = snapped.reshape(-1, 3, 3)

    # Drop triangles where any two verts coincide (zero-area after snapping)
    v0, v1, v2 = new_tris[:, 0], new_tris[:, 1], new_tris[:, 2]
    degen = (
        np.all(v0 == v1, axis=1) |
        np.all(v1 == v2, axis=1) |
        np.all(v0 == v2, axis=1)
    )
    kept = new_tris[~degen]

    stats = {
        "input_tris": int(len(tris_nx3x3)),
        "output_tris": int(len(kept)),
        "degenerate_dropped": int(degen.sum()),
        "ratio": float(len(kept) / max(len(tris_nx3x3), 1)),
        "grid_mm": float(grid_size),
    }
    return kept.astype(np.float32), stats


def _pick_grid_for_target(meshes, target_tris, max_iters=12):
    """Binary-search a grid size that yields ~target_tris total triangles."""
    all_flat = np.concatenate([m.reshape(-1, 3) for m in meshes], axis=0)
    bbox = all_flat.max(0) - all_flat.min(0)
    start = float(np.linalg.norm(bbox)) / 200.0 or 0.1

    lo, hi = start * 0.01, start * 10
    for _ in range(max_iters):
        mid = (lo + hi) / 2
        total = sum(cluster_simplify(m, mid)[1]["output_tris"] for m in meshes)
        if total > target_tris * 1.05:
            lo = mid
        elif total < target_tris * 0.95:
            hi = mid
        else:
            return mid
    return (lo + hi) / 2


def main():
    p = argparse.ArgumentParser(description="Simplify a .stl or .3mf mesh (vertex clustering).")
    p.add_argument("input")
    p.add_argument("output")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ratio", type=float,
                   help="Target ratio of output to input triangles (e.g. 0.5 = halve).")
    g.add_argument("--max-tris", type=int,
                   help="Target maximum triangle count across all meshes.")
    g.add_argument("--grid", type=float,
                   help="Explicit grid cell size in mm (advanced).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.input):
        msg = f"File not found: {args.input}"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    meshes = load_meshes(args.input)
    if not meshes:
        msg = "No mesh data in input"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(2)

    total_in = sum(len(m) for m in meshes)

    # Resolve grid size
    if args.grid:
        grid = args.grid
    elif args.max_tris:
        grid = _pick_grid_for_target(meshes, args.max_tris)
    elif args.ratio:
        grid = _pick_grid_for_target(meshes, int(total_in * args.ratio))
    else:
        # Sensible default: halve
        grid = _pick_grid_for_target(meshes, int(total_in * 0.5))

    # Apply
    new_meshes = []
    per_mesh = []
    for m in meshes:
        kept, st = cluster_simplify(m, grid)
        new_meshes.append(kept)
        per_mesh.append(st)
    total_out = sum(len(m) for m in new_meshes)

    # Write output. STL is a flat soup; 3MF re-builds a minimal structure.
    ext_out = os.path.splitext(args.output)[1].lower()
    if ext_out == ".stl":
        merged = np.concatenate(new_meshes, axis=0) if new_meshes else np.zeros((0, 3, 3), dtype=np.float32)
        write_stl_binary(args.output, merged,
                         header_text=f"Simplified by 3mf-stl-editor ({total_out}/{total_in} tris)")
    elif ext_out == ".3mf":
        # Reuse convert.py's builder so the output is a clean minimal 3MF
        from convert import build_3mf_model_xml, write_3mf
        model_xml = build_3mf_model_xml(new_meshes,
                                        names=[f"mesh_{i+1}_simplified" for i in range(len(new_meshes))])
        write_3mf(args.output, model_xml)
    else:
        msg = f"Unknown output extension '{ext_out}'"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    summary = {
        "ok": True,
        "output": args.output,
        "grid_mm": float(grid),
        "total_input_tris": int(total_in),
        "total_output_tris": int(total_out),
        "reduction": float(1 - total_out / max(total_in, 1)),
        "per_mesh": per_mesh,
    }
    if args.json:
        print(json.dumps(summary))
    else:
        print(f"Simplified {args.input} -> {args.output}")
        print(f"Grid size : {grid:.3f} mm")
        print(f"Triangles : {total_in:,} -> {total_out:,} ({summary['reduction']*100:.1f}% reduction)")
    sys.exit(0)


if __name__ == "__main__":
    main()
