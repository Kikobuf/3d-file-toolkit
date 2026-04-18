#!/usr/bin/env python3
"""
Detect the software that produced a 3D file and report compatibility
gotchas with other CAD / slicer / modeling programs.

Usage:
    compat.py <file>                # pretty report
    compat.py <file> --json         # machine-readable
    compat.py <file> --target bambu_studio   # will this open cleanly in X?

The check covers:
  * STL/3MF/OBJ/PLY/GLB — which writer produced the file (header strings,
    metadata fields, ZIP entry patterns, comment lines)
  * 3MF extensions in use (Production, Beam Lattice, Bambu custom)
  * Manifold / watertightness heuristics
  * Triangle / vertex count relative to known software limits
  * Unit system and scale (a 0.001-unit file from a CAD program probably needs
    a 1000× scale before slicing)
  * Suggests which target programs will open it cleanly
"""
import sys, os, argparse, json, zipfile, re, struct
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
from _formats import sniff_format, read
from _3mf_reader import read_3mf_metadata, NS


# Known target programs + their capabilities
TARGETS = {
    "bambu_studio":   {"name": "Bambu Studio",   "reads": {"stl","3mf","obj","step","amf"}, "prefers": "3mf"},
    "orca_slicer":    {"name": "Orca Slicer",    "reads": {"stl","3mf","obj","step"},        "prefers": "3mf"},
    "prusaslicer":    {"name": "PrusaSlicer",    "reads": {"stl","3mf","obj","amf","step"},  "prefers": "3mf"},
    "cura":           {"name": "UltiMaker Cura", "reads": {"stl","3mf","obj","x3d","amf"},   "prefers": "stl"},
    "simplify3d":     {"name": "Simplify3D",     "reads": {"stl","obj"},                     "prefers": "stl"},
    "fusion360":      {"name": "Fusion 360",     "reads": {"stl","obj","step","stp","iges","igs","sldprt","ipt","3mf"}, "prefers": "step"},
    "solidworks":     {"name": "SolidWorks",     "reads": {"step","stp","iges","igs","stl","sldprt"}, "prefers": "step"},
    "freecad":        {"name": "FreeCAD",        "reads": {"step","stp","iges","igs","stl","obj","3mf","ply","glb"}, "prefers": "step"},
    "blender":        {"name": "Blender",        "reads": {"stl","obj","ply","glb","gltf","fbx","dae"}, "prefers": "glb"},
    "meshlab":        {"name": "MeshLab",        "reads": {"stl","obj","ply","3mf","glb"},   "prefers": "ply"},
    "tinkercad":      {"name": "Tinkercad",      "reads": {"stl","obj","svg"},               "prefers": "stl"},
    "onshape":        {"name": "Onshape",        "reads": {"step","stp","iges","igs","stl","obj","sldprt"}, "prefers": "step"},
    "printables":     {"name": "Printables.com upload", "reads": {"stl","3mf","obj"},        "prefers": "3mf"},
    "makerworld":     {"name": "MakerWorld upload",     "reads": {"3mf","stl"},              "prefers": "3mf"},
    "thingiverse":    {"name": "Thingiverse upload",    "reads": {"stl","obj","3mf","amf"},  "prefers": "stl"},
}


# ──────────────────────────────────────────────────────────────────────────────
#  Source detection per format
# ──────────────────────────────────────────────────────────────────────────────

_SOURCE_PATTERNS_STL = [
    (re.compile(rb"Bambu"),                    "Bambu Studio"),
    (re.compile(rb"BambuStudio"),              "Bambu Studio"),
    (re.compile(rb"Orca"),                     "Orca Slicer"),
    (re.compile(rb"PrusaSlicer"),              "PrusaSlicer"),
    (re.compile(rb"Cura"),                     "UltiMaker Cura"),
    (re.compile(rb"SOLIDWORKS", re.IGNORECASE),"SolidWorks"),
    (re.compile(rb"Fusion",   re.IGNORECASE),  "Fusion 360"),
    (re.compile(rb"MeshLab",  re.IGNORECASE),  "MeshLab"),
    (re.compile(rb"Blender",  re.IGNORECASE),  "Blender"),
    (re.compile(rb"FreeCAD",  re.IGNORECASE),  "FreeCAD"),
    (re.compile(rb"OpenSCAD", re.IGNORECASE),  "OpenSCAD"),
    (re.compile(rb"Tinkercad"),                "Tinkercad"),
    (re.compile(rb"3mf-stl-editor"),           "3mf-stl-editor (this tool)"),
]


def detect_stl(path):
    """Detect the STL writer by sniffing the 80-byte header."""
    with open(path, "rb") as f:
        header = f.read(80)
    for pat, name in _SOURCE_PATTERNS_STL:
        if pat.search(header):
            return {"writer": name, "header": header.decode("ascii", "replace").strip()}
    header_str = header.decode("ascii", "replace").strip("\x00 ").strip()
    return {"writer": "unknown", "header": header_str or "(empty/binary header)"}


def detect_3mf(path):
    """Scan 3MF ZIP entries + metadata + XML namespaces to identify slicer."""
    meta = read_3mf_metadata(path)
    report = {"writer": "unknown", "metadata_application": meta.get("Application", ""),
              "metadata": meta, "extensions": [], "zip_entries": []}

    with zipfile.ZipFile(path, "r") as zf:
        report["zip_entries"] = zf.namelist()
        app = meta.get("Application", "")

        # Strong signals from Application field
        if "BambuStudio" in app:    report["writer"] = "Bambu Studio"
        elif "OrcaSlicer" in app or "Orca" in app: report["writer"] = "Orca Slicer"
        elif "PrusaSlicer" in app or "Slic3r" in app: report["writer"] = "PrusaSlicer"
        elif "Cura" in app:         report["writer"] = "UltiMaker Cura"
        elif "Creality" in app:     report["writer"] = "Creality Print"
        elif "Fusion" in app:       report["writer"] = "Fusion 360"
        elif "3mf-stl-editor" in app: report["writer"] = "3mf-stl-editor (this tool)"

        # Fallback: infer from auxiliary files
        if report["writer"] == "unknown":
            names = " ".join(report["zip_entries"]).lower()
            if "bambu"   in names: report["writer"] = "Bambu Studio (inferred)"
            elif "orca"  in names: report["writer"] = "Orca Slicer (inferred)"
            elif "prusa" in names: report["writer"] = "PrusaSlicer (inferred)"
            elif "auxiliaries/" in names and "metadata/model_settings" in names:
                report["writer"] = "Bambu/Orca family (inferred from layout)"

        # Detect namespaces / extensions used in the main model
        main = next((n for n in report["zip_entries"] if n.endswith("3dmodel.model")), None)
        if main:
            with zf.open(main) as f:
                head = f.read(2048).decode("utf-8", "replace")
            if "xmlns:BambuStudio" in head: report["extensions"].append("bambu_studio_namespace")
            if "xmlns:p="         in head: report["extensions"].append("production")
            if "beamlattice"      in head.lower(): report["extensions"].append("beam_lattice")
            if "xmlns:m="         in head: report["extensions"].append("materials")
            if "slic3rpe"         in head.lower(): report["extensions"].append("prusaslicer_namespace")

    # Bambu-specific files
    bambu_files = [e for e in report["zip_entries"] if
                   e.startswith("Auxiliaries/") or e.startswith("Metadata/")]
    if bambu_files and report["writer"] == "unknown":
        report["writer"] = "Bambu/Orca family (inferred from aux files)"
    report["auxiliary_file_count"] = len(bambu_files)
    return report


def detect_obj(path):
    """OBJs have comment headers like `# Blender 3.6 ...` or `# Fusion 360 export`."""
    with open(path, "r", errors="ignore") as f:
        head_lines = [f.readline() for _ in range(20)]
    head = "\n".join(head_lines)
    for pat, name in _SOURCE_PATTERNS_STL:     # same patterns work
        if pat.search(head.encode("ascii", "replace")):
            return {"writer": name, "header": head.strip()[:200]}
    return {"writer": "unknown", "header": head.strip()[:200]}


def detect_ply(path):
    """PLY headers sometimes contain `comment VCGLIB generated` or similar."""
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head and len(head) < 4096:
            line = f.readline()
            if not line: break
            head += line
    txt = head.decode("ascii", "replace")
    for pat, name in _SOURCE_PATTERNS_STL:
        if pat.search(head):
            return {"writer": name, "header": txt[:300]}
    m = re.search(r"comment\s+(.*)", txt)
    return {"writer": m.group(1).strip()[:80] if m else "unknown", "header": txt[:300]}


def detect_glb(path):
    """GLB extras/asset.generator identifies the exporter."""
    with open(path, "rb") as f:
        f.read(12)          # header
        clen, _ = struct.unpack("<I4s", f.read(8))
        gltf = json.loads(f.read(clen))
    gen = gltf.get("asset", {}).get("generator", "")
    ver = gltf.get("asset", {}).get("version", "")
    return {"writer": gen or "unknown", "version": ver, "asset": gltf.get("asset", {})}


# ──────────────────────────────────────────────────────────────────────────────
#  Geometric sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def geometry_health(meshes):
    """Heuristic health checks. No proper non-manifold detection (that needs
    an edge-adjacency pass); just fast spot checks."""
    if not meshes:
        return {"ok": False, "error": "no mesh data"}

    all_flat = np.concatenate([m.reshape(-1,3) for m in meshes], axis=0)
    n_tris = sum(len(m) for m in meshes)
    n_verts_unique = len(np.unique(all_flat, axis=0))

    mins, maxs = all_flat.min(0), all_flat.max(0)
    dims = maxs - mins

    # Zero-area triangles
    zero_area = 0
    for m in meshes:
        v0, v1, v2 = m[:, 0], m[:, 1], m[:, 2]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        zero_area += int((areas < 1e-9).sum())

    # Scale hint: everything < 1mm or > 10m is probably in wrong units
    max_dim = float(dims.max()) if dims.max() > 0 else 0
    scale_hint = None
    if max_dim < 0.5:       scale_hint = "Very small — possibly in meters, try ×1000"
    elif max_dim > 5000:    scale_hint = "Very large — possibly in thousandths, try ×0.001"

    return {
        "triangles": int(n_tris),
        "unique_vertices": int(n_verts_unique),
        "duplication_ratio": float(n_tris * 3 / max(n_verts_unique, 1)),
        "dimensions_mm": [float(dims[0]), float(dims[1]), float(dims[2])],
        "zero_area_triangles": zero_area,
        "scale_warning": scale_hint,
        "is_likely_watertight": zero_area == 0 and n_verts_unique * 2 <= n_tris * 3,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Compatibility matrix
# ──────────────────────────────────────────────────────────────────────────────

def compat_matrix(sniff_info, detection, health, target_key=None):
    """Build a per-target compatibility report."""
    ext = sniff_info["ext"].lstrip(".")
    writer = detection.get("writer", "unknown") if detection else "unknown"
    has_bambu_ns   = "bambu_studio_namespace" in (detection.get("extensions") or []) if detection else False
    has_prusa_ns   = "prusaslicer_namespace"  in (detection.get("extensions") or []) if detection else False
    has_beam       = "beam_lattice"           in (detection.get("extensions") or []) if detection else False
    high_tri_count = health.get("triangles", 0) > 1_000_000 if health else False

    out = {}
    targets = [target_key] if target_key else list(TARGETS)
    for key in targets:
        t = TARGETS.get(key)
        if not t: continue
        notes = []
        verdict = "ok"

        if ext not in t["reads"]:
            verdict = "blocked"
            notes.append(f"Does not read .{ext} — convert to .{t['prefers']} first.")
        else:
            if has_bambu_ns and key not in ("bambu_studio", "orca_slicer"):
                notes.append("Uses Bambu-specific XML namespace; non-Bambu programs will ignore print settings and some object metadata but geometry will open fine.")
            if has_prusa_ns and key not in ("prusaslicer",):
                notes.append("Uses PrusaSlicer namespace; other slicers will ignore Prusa-specific metadata.")
            if has_beam:
                verdict = "warn"
                notes.append("Contains 3MF Beam Lattice extension — only a few tools render this correctly (Bambu/Orca/Prusa do; Cura, Tinkercad don't).")
            if high_tri_count and key in ("tinkercad",):
                verdict = "warn"
                notes.append(f"{health['triangles']:,} triangles may be too heavy for {t['name']} — simplify to <200k first.")
            if t["prefers"] != ext and ext in ("stl", "obj") and key in ("fusion360","solidworks","onshape"):
                notes.append(f"{t['name']} can import .{ext} but it'll arrive as a mesh body (not a parametric solid). For editable geometry, export STEP.")
            if ext == "3mf" and key in ("thingiverse",):
                notes.append("Thingiverse accepts 3MF but many users expect STL — consider uploading both.")

        out[key] = {"name": t["name"], "verdict": verdict, "notes": notes,
                    "input_ext": ext, "preferred_ext": t["prefers"]}
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def run(path, target=None):
    info = sniff_format(path)
    if info["kind"] == "cad":
        return {
            "ok": True, "path": path, "format": info["detail"],
            "kind": "cad",
            "message": ("This is a CAD native format. Export to STL / OBJ / 3MF "
                        "from your CAD program to use with 3D print tooling."),
            "suggested_targets": ["fusion360", "solidworks", "freecad", "onshape"],
        }

    ext = info["ext"]
    detection = None
    if ext == ".stl": detection = detect_stl(path)
    elif ext == ".3mf": detection = detect_3mf(path)
    elif ext == ".obj": detection = detect_obj(path)
    elif ext == ".ply": detection = detect_ply(path)
    elif ext == ".glb": detection = detect_glb(path)

    # Geometry check (may be slow on huge files; skip if user asked targets only)
    try:
        meshes = read(path)
        health = geometry_health(meshes)
    except Exception as e:
        health = {"error": str(e)}

    compat = compat_matrix(info, detection, health, target_key=target)

    # Strip big arrays out of detection for the summary
    det_small = dict(detection or {})
    det_small.pop("zip_entries", None)
    # Only keep truncated metadata for 3MF
    if "metadata" in det_small:
        m = det_small["metadata"]
        keep = ("Title","Designer","Application","License","CreationDate","ModificationDate")
        det_small["metadata"] = {k: v[:200] for k, v in m.items() if k in keep}

    return {
        "ok": True,
        "path": path,
        "format": info["detail"],
        "kind": info["kind"],
        "size_bytes": info["size_bytes"],
        "detected": det_small,
        "health": health,
        "compat": compat,
    }


def print_pretty(r):
    print(f"File         : {r['path']}")
    print(f"Format       : {r['format']}")
    print(f"Size         : {r.get('size_bytes','?'):,} bytes")
    if r.get("kind") == "cad":
        print(f"\n{r['message']}")
        return

    det = r.get("detected") or {}
    print(f"\nWriter       : {det.get('writer','unknown')}")
    if det.get("extensions"):
        print(f"Extensions   : {', '.join(det['extensions'])}")
    if det.get("metadata"):
        for k, v in det["metadata"].items():
            print(f"  {k}: {v}")

    h = r.get("health") or {}
    if "error" not in h:
        print(f"\nTriangles    : {h['triangles']:,}")
        print(f"Unique verts : {h['unique_vertices']:,}")
        d = h["dimensions_mm"]
        print(f"Dimensions   : {d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f} mm")
        if h.get("zero_area_triangles", 0):
            print(f"Zero-area tris: {h['zero_area_triangles']:,} (bad — run simplify or a repair tool)")
        if h.get("scale_warning"):
            print(f"Scale note   : {h['scale_warning']}")
        print(f"Watertight?  : {'likely yes' if h.get('is_likely_watertight') else 'maybe not'}")

    print("\nCompatibility:")
    for key, c in (r.get("compat") or {}).items():
        tag = {"ok":"✓", "warn":"!", "blocked":"✗"}.get(c["verdict"], "?")
        print(f"  {tag} {c['name']:24s} ({c['verdict']})")
        for n in c["notes"]:
            print(f"      └─ {n}")


def main():
    p = argparse.ArgumentParser(description="Detect file source software + compatibility check")
    p.add_argument("path")
    p.add_argument("--target", help="Only check one target (e.g. bambu_studio, fusion360)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--list-targets", action="store_true")
    args = p.parse_args()

    if args.list_targets:
        for k, v in TARGETS.items():
            print(f"  {k:18s} {v['name']}  (reads: {', '.join(sorted(v['reads']))})")
        return

    if not os.path.exists(args.path):
        msg = f"File not found: {args.path}"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    report = run(args.path, target=args.target)
    if args.json:
        print(json.dumps(report, default=str))
    else:
        print_pretty(report)
    sys.exit(0)


if __name__ == "__main__":
    main()
