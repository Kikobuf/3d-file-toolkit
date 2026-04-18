#!/usr/bin/env python3
"""
Inspect a .3mf or .stl file: geometry stats + 3MF metadata.

Fixes vs v1:
  * HTML-escaped metadata values (Bambu's <Description> is 2 KB of &lt;p&gt;...)
    are now decoded and truncated in pretty-print mode. Full text is kept in
    --json mode.
  * Vertex counts now reflect unique positions, not N*3 triangle-soup counts.
  * Much faster parsing (uses shared _3mf_reader which regex-scans raw bytes).
  * --json output for agent consumption.
"""
import sys, os, argparse, json, html
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _3mf_reader import (
    read_stl, read_3mf, read_3mf_metadata, tag, NS,
)
import xml.etree.ElementTree as ET
import zipfile


PRINT_HINT_KEYS = ("print", "layer", "infill", "support", "material", "nozzle", "filament", "bed_temp")


def _truncate(s, n=140):
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _decode_html(s):
    """Decode possibly-double-encoded HTML entities, strip tags for preview."""
    if not s:
        return ""
    prev = None
    out = s
    # Unescape up to 2 passes (Bambu double-encodes, e.g. &amp;lt;)
    for _ in range(2):
        new = html.unescape(out)
        if new == prev:
            break
        prev = out = new
    # Strip HTML tags for readability
    import re
    out = re.sub(r"<[^>]+>", "", out)
    # Collapse whitespace
    out = " ".join(out.split())
    return out


def inspect_stl(path):
    tris = read_stl(path)[0]
    flat = tris.reshape(-1, 3)
    unique = np.unique(flat, axis=0)
    mins, maxs = flat.min(0), flat.max(0)
    dims = maxs - mins
    center = (mins + maxs) / 2

    return {
        "format": "stl",
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "triangles": int(len(tris)),
        "vertices_unique": int(len(unique)),
        "vertices_total": int(len(flat)),
        "dimensions_mm": [float(dims[0]), float(dims[1]), float(dims[2])],
        "bbox_min": [float(mins[0]), float(mins[1]), float(mins[2])],
        "bbox_max": [float(maxs[0]), float(maxs[1]), float(maxs[2])],
        "center": [float(center[0]), float(center[1]), float(center[2])],
    }


def inspect_3mf(path):
    meta = read_3mf_metadata(path)
    meshes = read_3mf(path)

    # Per-mesh and overall geometry
    mesh_infos = []
    all_verts_list = []
    for i, m in enumerate(meshes):
        flat = m.reshape(-1, 3)
        mins = flat.min(0); maxs = flat.max(0); dims = maxs - mins
        mesh_infos.append({
            "index": i,
            "triangles": int(len(m)),
            "vertices_total": int(len(flat)),
            "dimensions_mm": [float(dims[0]), float(dims[1]), float(dims[2])],
            "bbox_min": [float(mins[0]), float(mins[1]), float(mins[2])],
            "bbox_max": [float(maxs[0]), float(maxs[1]), float(maxs[2])],
        })
        all_verts_list.append(flat)

    overall_dims = None
    if all_verts_list:
        av = np.concatenate(all_verts_list, axis=0)
        overall_dims = (av.max(0) - av.min(0)).tolist()

    # ZIP entry inventory
    with zipfile.ZipFile(path, "r") as zf:
        zip_entries = [{"name": n, "size": zf.getinfo(n).file_size} for n in zf.namelist()]

    print_settings = {k: v for k, v in meta.items()
                      if any(h in k.lower() for h in PRINT_HINT_KEYS)}

    # Identify a thumbnail
    thumb = None
    for n in (e["name"] for e in zip_entries):
        if "thumbnail" in n.lower() and n.lower().endswith((".png", ".jpg", ".webp")):
            thumb = n
            break

    return {
        "format": "3mf",
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "meshes": mesh_infos,
        "overall_dimensions_mm": overall_dims,
        "metadata": meta,
        "print_settings": print_settings,
        "zip_entries": zip_entries,
        "thumbnail_entry": thumb,
    }


def print_pretty(info):
    print(f"Format       : {info['format'].upper()}")
    print(f"File size    : {info['file_size_bytes']:,} bytes")

    if info["format"] == "stl":
        d = info["dimensions_mm"]
        print(f"Triangles    : {info['triangles']:,}")
        print(f"Vertices     : {info['vertices_unique']:,} unique ({info['vertices_total']:,} total)")
        print(f"Dimensions   : {d[0]:.3f} x {d[1]:.3f} x {d[2]:.3f} mm")
        print(f"Bbox min     : ({info['bbox_min'][0]:.3f}, {info['bbox_min'][1]:.3f}, {info['bbox_min'][2]:.3f})")
        print(f"Bbox max     : ({info['bbox_max'][0]:.3f}, {info['bbox_max'][1]:.3f}, {info['bbox_max'][2]:.3f})")
        return

    # 3MF
    n_entries = len(info["zip_entries"])
    shown = ", ".join(e["name"] for e in info["zip_entries"][:6])
    print(f"ZIP entries  : {n_entries} ({shown}{'…' if n_entries > 6 else ''})")
    if info["thumbnail_entry"]:
        print(f"Thumbnail    : {info['thumbnail_entry']}")

    if info["metadata"]:
        print("\n--- Metadata ---")
        for k, v in info["metadata"].items():
            # Decode & truncate long HTML blobs
            cleaned = _decode_html(v) if any(h in v for h in ("&lt;", "&amp;", "<p>", "<span")) else v
            print(f"  {k}: {_truncate(cleaned, 140)}")

    print(f"\nObjects (world-space, transforms applied): {len(info['meshes'])}")
    for m in info["meshes"]:
        d = m["dimensions_mm"]
        print(f"  [{m['index']}] {m['triangles']:,} tris, {d[0]:.2f}x{d[1]:.2f}x{d[2]:.2f} mm")

    if info["overall_dimensions_mm"]:
        d = info["overall_dimensions_mm"]
        print(f"\nOverall bbox : {d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f} mm")

    if info["print_settings"]:
        print("\n--- Print Settings (hint) ---")
        for k, v in info["print_settings"].items():
            print(f"  {k}: {_truncate(v, 100)}")


def main():
    p = argparse.ArgumentParser(description="Inspect .stl or .3mf geometry + metadata")
    p.add_argument("path")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = p.parse_args()

    if not os.path.exists(args.path):
        msg = f"File not found: {args.path}"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    ext = os.path.splitext(args.path)[1].lower()
    if ext == ".stl":
        info = inspect_stl(args.path)
    elif ext == ".3mf":
        info = inspect_3mf(args.path)
    else:
        msg = f"Unknown extension '{ext}' (expected .stl or .3mf)"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    info["ok"] = True
    if args.json:
        print(json.dumps(info, default=str))
    else:
        print_pretty(info)


if __name__ == "__main__":
    main()
