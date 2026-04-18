#!/usr/bin/env python3
"""Edit metadata fields in a 3MF file.

Fixes vs v1:
  * --remove KEY support (e.g. strip DesignerUserId before sharing).
  * --json mode for both --list and set/remove summaries.
  * Warns but still writes if input isn't a 3MF (STL has no metadata).
"""
import sys, os, zipfile, argparse, json
import xml.etree.ElementTree as ET

NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
ET.register_namespace("", NS)
tag = lambda t: f"{{{NS}}}{t}"


def _read_main_model(path):
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        mp = next((n for n in names if n.endswith("3dmodel.model")), None)
        if not mp: return None, None, None
        with zf.open(mp) as f:
            return mp, f.read(), names


def _write_with_patched_model(inp, out, model_path, new_model_bytes):
    with zipfile.ZipFile(inp, "r") as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == model_path:
                zout.writestr(item, new_model_bytes)
            else:
                zout.writestr(item, zin.read(item.filename))


def main():
    p = argparse.ArgumentParser(description="Edit 3MF metadata — single file or batch")
    p.add_argument("input", help="Input .3mf file, or a glob pattern with --batch (e.g. '*.3mf')")
    p.add_argument("output", nargs="?", help="Output path (required for single file; ignored with --batch)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="Set a metadata field. Repeatable.")
    p.add_argument("--remove", action="append", metavar="KEY",
                   help="Remove a metadata field. Repeatable.")
    p.add_argument("--template", metavar="FILE",
                   help="JSON file containing {KEY: VALUE} pairs to set (merged with --set/--remove).")
    p.add_argument("--batch", action="store_true",
                   help="Process all .3mf files matching the input glob in-place.")
    p.add_argument("--list", action="store_true", help="List current metadata and exit")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    # Load template if provided
    template_updates = {}
    if args.template:
        if not os.path.exists(args.template):
            msg = f"Template not found: {args.template}"
            print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
            sys.exit(1)
        with open(args.template) as f:
            template_updates = json.load(f)

    # Resolve input files
    if args.batch:
        import glob as _glob
        paths = _glob.glob(args.input, recursive=True)
        if not paths:
            msg = f"No files matched: {args.input}"
            print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
            sys.exit(1)
        results = []
        for inp in sorted(paths):
            out = inp  # in-place for batch
            r = _process_one(inp, out, args, template_updates, in_place=True)
            results.append(r)
            if not args.json:
                status = "ok" if r["ok"] else f"ERROR: {r.get('error','?')}"
                print(f"  {inp}: {status}")
        if args.json:
            print(json.dumps({"ok": True, "processed": len(results), "results": results}))
        else:
            ok = sum(1 for r in results if r["ok"])
            print(f"Batch complete: {ok}/{len(results)} files updated")
        sys.exit(0)

    # Single file
    if not os.path.exists(args.input):
        msg = f"File not found: {args.input}"
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    if args.list:
        mp, content, _ = _read_main_model(args.input)
        if mp is None:
            print(json.dumps({"ok": False, "error": "No 3dmodel.model found"}) if args.json
                  else "ERROR: No 3dmodel.model found")
            sys.exit(1)
        root = ET.fromstring(content)
        meta = {m.get("name",""): (m.text or "") for m in root.findall(tag("metadata"))}
        if args.json:
            print(json.dumps({"ok": True, "metadata": meta}))
        else:
            print("Current metadata:")
            if not meta: print("  (none)")
            for k, v in meta.items():
                print(f"  {k}: {v}")
        sys.exit(0)

    if not args.set and not args.remove and not template_updates:
        msg = "Nothing to do — pass --set, --remove, --template, or --list."
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    if not args.output:
        msg = "Output path required when modifying metadata (or use --batch for in-place)."
        print(json.dumps({"ok": False, "error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    summary = _process_one(args.input, args.output, args, template_updates, in_place=False)
    if args.json:
        print(json.dumps(summary))
    else:
        for k in summary.get("removed",[]): print(f"  Removed: {k}")
        for k in summary.get("missing",[]): print(f"  Skipped (not present): {k}")
        for k in summary.get("added",[]):   print(f"  Added: {k}")
        for k in summary.get("set",[]):     print(f"  Updated: {k}")
        print(f"Saved: {args.output}")
    sys.exit(0 if summary.get("ok") else 2)


def _process_one(inp, out, args, template_updates, in_place=False):
    """Process a single file — shared by single and batch modes."""
    import tempfile
    mp, content, _ = _read_main_model(inp)
    if mp is None:
        return {"ok": False, "path": inp, "error": "No 3dmodel.model found"}

    root = ET.fromstring(content)
    existing = {m.get("name",""): m for m in root.findall(tag("metadata"))}
    changed = {"set": [], "added": [], "removed": [], "missing": []}

    # Merge template + --set
    updates = dict(template_updates)
    for s in (args.set or []):
        if "=" not in s:
            return {"ok": False, "path": inp, "error": f"--set must be KEY=VALUE, got: {s!r}"}
        k, v = s.split("=", 1)
        updates[k.strip()] = v.strip()

    # Removes
    for k in (args.remove or []):
        if k in existing:
            root.remove(existing[k]); del existing[k]
            changed["removed"].append(k)
        else:
            changed["missing"].append(k)

    # Sets
    for k, v in updates.items():
        if k in existing:
            existing[k].text = str(v); changed["set"].append(k)
        else:
            m = ET.Element(tag("metadata"), name=k); m.text = str(v)
            res = root.find(tag("resources"))
            if res is not None: root.insert(list(root).index(res), m)
            else: root.append(m)
            existing[k] = m; changed["added"].append(k)

    new_xml = ET.tostring(root, encoding="unicode", xml_declaration=False)
    new_bytes = ('<?xml version="1.0" encoding="UTF-8"?>\n' + new_xml).encode("utf-8")

    # For in-place batch: write to temp then replace
    if in_place:
        tmp = inp + ".tmp"
        _write_with_patched_model(inp, tmp, mp, new_bytes)
        os.replace(tmp, out)
    else:
        _write_with_patched_model(inp, out, mp, new_bytes)

    return {"ok": True, "path": inp, "output": out, **changed}


if __name__ == "__main__":
    main()
