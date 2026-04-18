#!/usr/bin/env python3
"""Merge multiple STL/3MF files into a single output file.

Fixes vs v1:
  * 3MF inputs with Bambu-style external sub-files (3D/Objects/object_N.model)
    are now read via the shared transform-aware reader; previously such inputs
    produced empty output.
  * STL output uses the validated 80-byte header writer.
  * --json mode.
"""
import sys, os, zipfile, argparse, datetime, json
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _3mf_reader import NS, read_stl, write_stl_binary, read_3mf

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""
RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model" Id="rel0"/>
</Relationships>"""


def _load_any_named(path):
    """Return [(name, tris_nx3x3)]. 3MFs contribute one entry per mesh."""
    ext = os.path.splitext(path)[1].lower()
    stem = os.path.splitext(os.path.basename(path))[0]
    if ext == ".stl":
        return [(stem, read_stl(path)[0])]
    if ext == ".3mf":
        named = read_3mf(path, with_names=True)
        # read_3mf returns List[(name, tris)] when with_names=True
        return [(f"{stem}__{nm}", t) for nm, t in named] if named else []
    raise ValueError(f"Unknown extension: {ext}")


def _write_merged_3mf(path, named_meshes):
    root = ET.Element(f"{{{NS}}}model", attrib={"unit":"millimeter","xml:lang":"en-US"})
    today = datetime.date.today().isoformat()
    for k, v in [("Title", "Merged"), ("Application", "3mf-stl-editor"), ("CreationDate", today)]:
        m = ET.SubElement(root, f"{{{NS}}}metadata", name=k); m.text = v
    res = ET.SubElement(root, f"{{{NS}}}resources")
    build = ET.SubElement(root, f"{{{NS}}}build")

    for idx, (name, tris) in enumerate(named_meshes):
        obj = ET.SubElement(res, f"{{{NS}}}object", id=str(idx+1), type="model", name=name)
        mesh_el = ET.SubElement(obj, f"{{{NS}}}mesh")
        ve = ET.SubElement(mesh_el, f"{{{NS}}}vertices")
        te = ET.SubElement(mesh_el, f"{{{NS}}}triangles")

        flat = tris.reshape(-1, 3).astype(np.float32)
        unique, inverse = np.unique(flat, axis=0, return_inverse=True)
        tri_idx = inverse.reshape(-1, 3)
        for v in unique:
            ET.SubElement(ve, f"{{{NS}}}vertex",
                          x=f"{v[0]:.6f}", y=f"{v[1]:.6f}", z=f"{v[2]:.6f}")
        for t in tri_idx:
            ET.SubElement(te, f"{{{NS}}}triangle",
                          v1=str(int(t[0])), v2=str(int(t[1])), v3=str(int(t[2])))
        ET.SubElement(build, f"{{{NS}}}item", objectid=str(idx+1))

    xml = ET.tostring(root, encoding="unicode", xml_declaration=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("3D/3dmodel.model", '<?xml version="1.0" encoding="UTF-8"?>\n' + xml)


def main():
    p = argparse.ArgumentParser(description="Merge multiple STL/3MF into one file.")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    for f in args.inputs:
        if not os.path.exists(f):
            msg = f"File not found: {f}"
            print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
            sys.exit(1)

    all_meshes = []
    per_file = []
    for f in args.inputs:
        ms = _load_any_named(f)
        all_meshes.extend(ms)
        per_file.append({"path": f, "meshes": len(ms), "triangles": sum(len(t) for _, t in ms)})

    if not all_meshes:
        msg = "No meshes found in any input"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(2)

    ext_out = os.path.splitext(args.output)[1].lower()
    if ext_out == ".3mf":
        _write_merged_3mf(args.output, all_meshes)
    elif ext_out == ".stl":
        merged = np.concatenate([t for _, t in all_meshes], axis=0)
        write_stl_binary(args.output, merged, header_text=f"Merged by 3mf-stl-editor ({len(all_meshes)} meshes)")
    else:
        msg = f"Unknown output extension '{ext_out}' (expected .stl or .3mf)"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    total = sum(len(t) for _, t in all_meshes)
    summary = {
        "ok": True, "output": args.output,
        "format_out": ext_out[1:],
        "input_files": per_file,
        "total_meshes": len(all_meshes),
        "total_triangles": int(total),
    }
    if args.json:
        print(json.dumps(summary))
    else:
        for pf in per_file:
            print(f"  Loaded {pf['meshes']} mesh(es), {pf['triangles']:,} tris from {os.path.basename(pf['path'])}")
        print(f"Merged {len(all_meshes)} mesh(es), {total:,} triangles -> {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
