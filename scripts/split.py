#!/usr/bin/env python3
"""Extract one mesh from a multi-mesh 3MF (or list all meshes).

Fixes vs v1:
  * Previously walked only the main `3D/3dmodel.model` object list; for Bambu
    Studio files where each logical object is a <component> pointing to a
    separate sub-file, listing returned a single meaningless entry. Now uses
    the shared reader which flattens the full build hierarchy.
  * STL output uses the validated 80-byte header writer.
  * --json mode for both --list and extract.
"""
import sys, os, zipfile, argparse, datetime, json
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _3mf_reader import NS, read_3mf, write_stl_binary

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""
RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model" Id="rel0"/>
</Relationships>"""


def _write_single_3mf(path, name, tris):
    root = ET.Element(f"{{{NS}}}model", attrib={"unit":"millimeter","xml:lang":"en-US"})
    for k, v in [("Title", name), ("Application", "3mf-stl-editor"),
                 ("CreationDate", datetime.date.today().isoformat())]:
        m = ET.SubElement(root, f"{{{NS}}}metadata", name=k); m.text = v
    res = ET.SubElement(root, f"{{{NS}}}resources")
    build = ET.SubElement(root, f"{{{NS}}}build")
    obj = ET.SubElement(res, f"{{{NS}}}object", id="1", type="model", name=name)
    mesh_el = ET.SubElement(obj, f"{{{NS}}}mesh")
    ve = ET.SubElement(mesh_el, f"{{{NS}}}vertices")
    te = ET.SubElement(mesh_el, f"{{{NS}}}triangles")

    flat = tris.reshape(-1, 3).astype(np.float32)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    tri_idx = inverse.reshape(-1, 3)
    for v in unique:
        ET.SubElement(ve, f"{{{NS}}}vertex", x=f"{v[0]:.6f}", y=f"{v[1]:.6f}", z=f"{v[2]:.6f}")
    for t in tri_idx:
        ET.SubElement(te, f"{{{NS}}}triangle", v1=str(int(t[0])), v2=str(int(t[1])), v3=str(int(t[2])))
    ET.SubElement(build, f"{{{NS}}}item", objectid="1")

    xml = ET.tostring(root, encoding="unicode", xml_declaration=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("3D/3dmodel.model", '<?xml version="1.0" encoding="UTF-8"?>\n' + xml)


def main():
    p = argparse.ArgumentParser(description="Extract one mesh from a multi-mesh 3MF")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="Output path (.stl or .3mf); required unless --list")
    p.add_argument("--object-index", type=int, default=0, help="0-based mesh index (see --list)")
    p.add_argument("--list", action="store_true", help="List meshes and exit (no output)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.input):
        msg = f"File not found: {args.input}"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    named = read_3mf(args.input, with_names=True)

    if args.list or not args.output:
        entries = []
        for i, (name, tris) in enumerate(named):
            flat = tris.reshape(-1, 3)
            dims = (flat.max(0) - flat.min(0)).tolist()
            entries.append({"index": i, "name": name, "triangles": int(len(tris)), "dimensions_mm": dims})
        if args.json:
            print(json.dumps({"ok": True, "meshes": entries}))
        else:
            print(f"Meshes in {os.path.basename(args.input)}:")
            for e in entries:
                d = e["dimensions_mm"]
                print(f"  [{e['index']}] {e['name']!r}: {e['triangles']:,} tris, {d[0]:.2f}x{d[1]:.2f}x{d[2]:.2f} mm")
            if not entries: print("  (none found)")
        sys.exit(0)

    idx = args.object_index
    if idx < 0 or idx >= len(named):
        msg = f"Mesh index {idx} out of range (0..{len(named)-1})"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    name, tris = named[idx]
    ext_out = os.path.splitext(args.output)[1].lower()

    if ext_out == ".stl":
        write_stl_binary(args.output, tris, header_text=f"Split from 3mf-stl-editor: {name}")
    elif ext_out == ".3mf":
        _write_single_3mf(args.output, name, tris)
    else:
        msg = f"Unknown output extension '{ext_out}'"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    summary = {"ok": True, "output": args.output, "name": name,
               "index": idx, "triangles": int(len(tris))}
    if args.json:
        print(json.dumps(summary))
    else:
        print(f"Extracted [{idx}] {name!r} -> {args.output} ({len(tris):,} tris)")
    sys.exit(0)


if __name__ == "__main__":
    main()
