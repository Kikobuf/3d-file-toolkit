#!/usr/bin/env python3
"""
Convert between .stl and .3mf formats.

Fixes vs v1:
  * STL header is now exactly 80 bytes (v1 wrote 79, causing uint32 count
    field misalignment and garbage triangle counts on output).
  * 3MF -> STL now applies the full build transform chain (previously it
    extracted mesh-local coordinates, producing STLs at origin instead of
    on the build plate).
  * Uses shared `_3mf_reader` module (no duplicated STL parser).
  * --json output mode.
"""
import sys, os, zipfile, json, argparse, datetime
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _3mf_reader import (
    NS, read_stl, write_stl_binary, read_3mf,
)

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""
RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model" Id="rel0"/>
</Relationships>"""


def build_3mf_model_xml(tris_per_mesh, names=None):
    """Build a minimal 3MF main model XML from a list of (N,3,3) triangle arrays.
    Each mesh becomes one <object> with inline <mesh>. Vertices are deduped
    per-mesh via numpy (avoids exploding file size for indexed formats)."""
    root = ET.Element(f"{{{NS}}}model", attrib={"unit":"millimeter","xml:lang":"en-US"})
    today = datetime.date.today().isoformat()
    for key, val in [("Application", "3mf-stl-editor"), ("CreationDate", today)]:
        m = ET.SubElement(root, f"{{{NS}}}metadata", name=key); m.text = val
    resources = ET.SubElement(root, f"{{{NS}}}resources")
    build_el = ET.SubElement(root, f"{{{NS}}}build")

    for idx, tris in enumerate(tris_per_mesh):
        name = (names[idx] if names and idx < len(names) else f"mesh_{idx+1}")
        obj = ET.SubElement(resources, f"{{{NS}}}object", id=str(idx+1), type="model", name=name)
        mesh_el  = ET.SubElement(obj, f"{{{NS}}}mesh")
        verts_el = ET.SubElement(mesh_el, f"{{{NS}}}vertices")
        tris_el  = ET.SubElement(mesh_el, f"{{{NS}}}triangles")

        # Dedupe verts with numpy: round to 6dp, then unique
        flat = tris.reshape(-1, 3)
        rounded = np.round(flat, 6)
        _, unique_idx, inverse = np.unique(
            rounded.view([('', rounded.dtype)] * 3).reshape(-1), return_index=True, return_inverse=True)
        unique_verts = flat[unique_idx]
        tri_indices = inverse.reshape(-1, 3)

        for v in unique_verts:
            ET.SubElement(verts_el, f"{{{NS}}}vertex",
                          x=f"{v[0]:.6f}", y=f"{v[1]:.6f}", z=f"{v[2]:.6f}")
        for t in tri_indices:
            ET.SubElement(tris_el, f"{{{NS}}}triangle",
                          v1=str(int(t[0])), v2=str(int(t[1])), v3=str(int(t[2])))

        ET.SubElement(build_el, f"{{{NS}}}item", objectid=str(idx+1))

    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def write_3mf(path, model_xml):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("3D/3dmodel.model", '<?xml version="1.0" encoding="UTF-8"?>\n' + model_xml)


def stl_to_3mf(inp, out):
    tris = read_stl(inp)[0]
    name = os.path.splitext(os.path.basename(inp))[0]
    model_xml = build_3mf_model_xml([tris], names=[name])
    write_3mf(out, model_xml)
    return {"ok": True, "format_in": "stl", "format_out": "3mf",
            "triangles": int(len(tris)), "output": out}


def mf3_to_stl(inp, out, merge=True):
    """Extract all meshes (with build transforms applied) and write to STL.
    STL is a flat triangle soup; if the 3MF has multiple meshes they're merged."""
    meshes = read_3mf(inp)
    if not meshes:
        return {"ok": False, "error": "No mesh data found in 3MF"}

    merged = np.concatenate(meshes, axis=0)
    write_stl_binary(out, merged, header_text=f"3mf-stl-editor convert from {os.path.basename(inp)}")
    return {
        "ok": True, "format_in": "3mf", "format_out": "stl",
        "input_meshes": len(meshes),
        "triangles": int(len(merged)),
        "output": out,
    }


def main():
    p = argparse.ArgumentParser(description="Convert between .stl / .3mf / .obj / .ply.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    args = p.parse_args()

    if not os.path.exists(args.input):
        msg = f"File not found: {args.input}"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    ext_in  = os.path.splitext(args.input)[1].lower()
    ext_out = os.path.splitext(args.output)[1].lower()

    # Legacy fast paths for the common cases (preserve existing behavior + tests)
    if ext_in == ".stl" and ext_out == ".3mf":
        summary = stl_to_3mf(args.input, args.output)
    elif ext_in == ".3mf" and ext_out == ".stl":
        summary = mf3_to_stl(args.input, args.output)
    elif ext_in == ext_out:
        msg = "Same input/output format — use transform.py for in-place edits."
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)
    else:
        # General path via _formats: supports OBJ, PLY, and any mesh ext pair.
        from _formats import read as read_any, write as write_any, MESH_EXTS
        if ext_in not in MESH_EXTS or ext_out not in MESH_EXTS:
            msg = f"Unsupported conversion: {ext_in} -> {ext_out}"
            print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
            sys.exit(1)
        try:
            meshes = read_any(args.input)
        except Exception as e:
            msg = str(e)
            print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
            sys.exit(1)
        write_any(args.output, meshes)
        total_tris = int(sum(len(m) for m in meshes)) if meshes else 0
        summary = {"ok": True, "format_in": ext_in.lstrip("."),
                   "format_out": ext_out.lstrip("."), "triangles": total_tris,
                   "input_meshes": len(meshes), "output": args.output}

    if args.json:
        print(json.dumps(summary))
    else:
        if summary.get("ok"):
            print(f"Converted {summary['format_in'].upper()} -> {summary['format_out'].upper()}: {summary['output']}")
            if "input_meshes" in summary and summary["input_meshes"] > 1:
                print(f"Merged {summary['input_meshes']} mesh(es) into one STL.")
            print(f"Triangles: {summary['triangles']:,}")
        else:
            print(f"ERROR: {summary.get('error','unknown')}")

    sys.exit(0 if summary.get("ok") else 2)


if __name__ == "__main__":
    main()
