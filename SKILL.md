---
name: 3d-file-toolkit
description: >
  Read, view, and edit 3D model files (.3mf, .stl, .obj, .ply, .glb). Use this
  skill whenever the user uploads, mentions, or asks about any 3D model file —
  even casually ("look at this model", "scale my print file", "what's in this
  3MF", "convert to STL"). Also covers detecting which CAD/slicer produced a
  file and checking compatibility with other programs (Bambu, Orca, Prusa,
  Cura, Fusion, SolidWorks, Blender, etc.), and generating new 3D models from
  text prompts or images (local primitive generation always works; API
  providers Meshy/Tripo/Replicate available with user-supplied keys). Covers:
  inspecting geometry + metadata, rendering inline 3D previews, thumbnails,
  scale/translate/rotate, simplify, merge/split, convert between formats,
  edit 3MF metadata, identify source software, check cross-program
  compatibility, and generate meshes from prompts. Always trigger this skill
  proactively when the user provides a 3D file or asks to generate one.
---

# 3D Model Toolkit (3MF / STL / OBJ / PLY / GLB)

A portable AI-agent skill for inspecting, editing, analyzing, and generating
3D model files. Works with any AI coding agent that can read a skill markdown
and shell out to Python (Claude Code, Claude.ai, Cursor, ChatGPT Code
Interpreter, Manus, Aider, etc.).

## Quick-start decision tree

| User wants to…                              | Script              |
|---------------------------------------------|---------------------|
| See what's in the file                      | `info.py`           |
| See a 3D render or thumbnail                | `render.py`         |
| Scale / move / rotate the mesh              | `transform.py`      |
| Reduce triangle count for a smaller file    | `simplify.py`       |
| Merge two meshes or split one               | `merge.py` / `split.py` |
| Change 3MF metadata or print settings       | `edit_metadata.py`  |
| Convert between .stl / .3mf / .obj / .ply   | `convert.py`        |
| Detect what made the file / compat check    | `compat.py`         |

Every script accepts `--json` for machine-parseable output and exits
`0` on success, `1` on bad arguments, `2` on operation failure.

## Environment

- **Python 3.8+** with **numpy** (required). No other dependencies.
- Scripts live in `${SKILL_DIR}/scripts/` (wherever this skill is installed).
  Invoke them by absolute path; they resolve their own dependencies via
  `sys.path`.
- All scripts operate on file paths provided as arguments. This skill makes
  no assumptions about upload/download directories — the host agent is
  responsible for file plumbing.

## File format primer

**STL** — pure triangle geometry, two flavors:
- *Binary STL*: 80-byte header + uint32 triangle count + 50 bytes per triangle.
- *ASCII STL*: text, starts with `solid`.

**3MF** — a ZIP archive containing:
- `3D/3dmodel.model` — main XML with object/component/build hierarchy.
- `[Content_Types].xml`, `_rels/.rels` — package metadata.
- **Often**: `3D/Objects/object_N.model` — sub-files holding the actual mesh
  geometry, referenced from the main model via `<component p:path="...">`.
  Bambu Studio, Orca Slicer, and PrusaSlicer all use this pattern. This skill
  handles it transparently — older 3MF tools that only read the main model
  see these files as empty.

## Supported formats

**Mesh formats — full read/write** (pure numpy + stdlib, no dependencies):
- `.stl`  — binary + ASCII
- `.3mf`  — ZIP + XML (handles Bambu/Orca/Prusa external sub-files)
- `.obj`  — Wavefront ASCII (geometry only; materials ignored)
- `.ply`  — binary LE + ASCII (geometry only)
- `.glb`  — GLTF binary (**read only** in this version)

**CAD native formats — detected but not editable** (require a CAD kernel):
- `.step` / `.stp` / `.iges` / `.igs` — ISO-standard CAD B-rep
- `.f3d` / `.f3z` — Fusion 360 native
- `.sldprt` — SolidWorks Part
- `.ipt` — Inventor Part

When the user passes a CAD file, `compat.py` identifies it and suggests
exporting to STL/OBJ/3MF from their CAD program first. Don't try to parse them.

## §View: Stats & Metadata

```bash
python3 ${SKILL_DIR}/scripts/info.py <file>
python3 ${SKILL_DIR}/scripts/info.py <file> --json
```

Reports file format, size, triangle count, unique-vertex count, world-space
bounding box, and (for 3MF) all `<metadata>` entries + any detected print
settings. HTML-encoded metadata values (common in Bambu files) are decoded
and truncated in pretty-print mode; full text is preserved in `--json`.

## §Identify: Source software + compatibility

```bash
python3 ${SKILL_DIR}/scripts/compat.py <file>
python3 ${SKILL_DIR}/scripts/compat.py <file> --target bambu_studio
python3 ${SKILL_DIR}/scripts/compat.py <file> --list-targets
```

Detects which program produced the file (Bambu Studio, Orca Slicer,
PrusaSlicer, Cura, Fusion 360, SolidWorks, Blender, FreeCAD, MeshLab,
OpenSCAD, Tinkercad, etc.) by sniffing STL headers, 3MF metadata + XML
namespaces, OBJ comments, PLY comments, and GLB `asset.generator`.

Then reports a compatibility matrix against major targets: which programs
will open it cleanly, which will warn, and which can't read the format at
all. Flags specific gotchas (Bambu-specific 3MF namespaces, 3MF Beam Lattice
extension, triangle counts too high for Tinkercad, etc.) with concrete
suggested fixes.

Use this whenever the user asks "will this open in X?" or "what software
made this file?".

## §View: 3D Render

```bash
# Interactive WebGL viewer — use --open so the user sees it right away
python3 ${SKILL_DIR}/scripts/render.py <file> <o>.html --open

# Static PNG thumbnail
python3 ${SKILL_DIR}/scripts/render.py <file> <o>.png
```

**IMPORTANT for agents**: Always surface the rendered file after creating it.
In Claude.ai use `present_files`. In Claude Code pass `--open` or open the file.
Never just print the output path and stop — the user needs to be able to see it.

Models display upright by default (Z-up matches standard 3MF/slicer convention).
Drag to rotate, scroll to zoom, shift+drag to pan.


## §Edit: Transforms

```bash
# Uniform scale
python3 ${SKILL_DIR}/scripts/transform.py <in> <out> --scale 2.0

# Per-axis scale
python3 ${SKILL_DIR}/scripts/transform.py <in> <out> --scale 1.0 2.0 1.0

# Translate (mm)
python3 ${SKILL_DIR}/scripts/transform.py <in> <out> --translate 10 0 -5

# Rotate around axis (degrees)
python3 ${SKILL_DIR}/scripts/transform.py <in> <out> --rotate-x 90
python3 ${SKILL_DIR}/scripts/transform.py <in> <out> --rotate-z 45

# Chain multiple ops (applied in order: scale → rotate → translate)
python3 ${SKILL_DIR}/scripts/transform.py <in> <out> \
    --scale 0.5 --rotate-z 90 --translate 0 0 10
```

Input and output extensions must match. For format conversion, use
`convert.py`. For 3MF files with external sub-models (Bambu/Orca/Prusa),
the transform is applied to every `.model` file in the archive, not just
the main one.

## §Edit: Merge & Split

```bash
# Merge multiple files into one
python3 ${SKILL_DIR}/scripts/merge.py f1.stl f2.3mf f3.stl -o merged.3mf

# List meshes in a multi-mesh 3MF
python3 ${SKILL_DIR}/scripts/split.py <in>.3mf --list

# Extract one mesh (0-based index)
python3 ${SKILL_DIR}/scripts/split.py <in>.3mf -o <out>.stl --object-index 0
```

Split extracts a single mesh from the build hierarchy (transforms applied).
Merge accepts mixed STL/3MF inputs and writes either format.

## §Edit: 3MF Metadata

```bash
# List current fields
python3 ${SKILL_DIR}/scripts/edit_metadata.py <file>.3mf --list

# Set / add fields
python3 ${SKILL_DIR}/scripts/edit_metadata.py <in>.3mf <out>.3mf \
    --set Title="My Part" \
    --set Designer="Enrico" \
    --set Description="v2 with thicker walls"

# Remove fields (useful before sharing — strip DesignerUserId, etc.)
python3 ${SKILL_DIR}/scripts/edit_metadata.py <in>.3mf <out>.3mf \
    --remove DesignerUserId --remove ProfileUserId
```

STL has no metadata; convert to 3MF first if you need it.

## §Edit: Simplify (reduce triangle count)

```bash
# Target a specific triangle count
python3 ${SKILL_DIR}/scripts/simplify.py <in> <out> --max-tris 20000

# Target a reduction ratio (0.5 = halve)
python3 ${SKILL_DIR}/scripts/simplify.py <in> <out> --ratio 0.3

# Explicit grid cell size (advanced)
python3 ${SKILL_DIR}/scripts/simplify.py <in> <out> --grid 0.5
```

Uses vertex clustering — fast (~3s on the 225k-tri benchy), lossy, preserves
overall shape but softens fine detail. For print-ready quality use a proper
slicer's "simplify" feature. For quick previews, LODs, or shipping smaller
files online, this is exactly the right trade-off.

## §Convert

```bash
# Any mesh format to any other
python3 ${SKILL_DIR}/scripts/convert.py <in>.stl  <out>.3mf
python3 ${SKILL_DIR}/scripts/convert.py <in>.3mf  <out>.obj
python3 ${SKILL_DIR}/scripts/convert.py <in>.obj  <out>.ply
python3 ${SKILL_DIR}/scripts/convert.py <in>.ply  <out>.stl
```

All pairwise conversions between `.stl`, `.3mf`, `.obj`, `.ply` are supported.
`.glb` is read-only (use it as a source, not a target). CAD native formats
(STEP / IGES / F3D / SLDPRT) can't be converted — export to a mesh format
from your CAD program first.

For 3MF→STL, the full build transform chain is applied and all meshes are
merged into one flat triangle soup (STL has no notion of separate objects).
If the source has multiple meshes, tell the user their logical grouping
will be lost.

## Host-specific integration notes

See `docs/AGENTS.md` for per-host integration tips (Claude Code,
Claude.ai, Cursor, ChatGPT, Manus, Aider). Short version:

- **File handling**: the scripts read/write arbitrary paths. Let your host's
  normal file-presentation mechanism handle the rest.
- **Machine output**: always pass `--json` when parsing results
  programmatically. Never scrape the pretty-print output.
- **Dependency check**: if a script fails with `ModuleNotFoundError: numpy`,
  run `pip install -r requirements.txt` (or `pip install numpy` for the core,
  plus `matplotlib` for PNG rendering).

## Error handling

- Missing input file → exits 1 with `{"ok": false, "error": "File not found: ..."}`.
- Unsupported format / bad args → exits 1.
- Transform produced zero vertices (usually means malformed input) → exits 2.
- Corrupt 3MF (missing `3D/3dmodel.model`) → exits 1 with hint to re-export
  from the slicer.
