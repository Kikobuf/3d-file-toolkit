# Changelog

## v2.1 — 2026-04

Broad-format and generative features.

### New formats
- **OBJ** (Wavefront ASCII) — read + write, n-gon fan triangulation.
- **PLY** (binary LE + ASCII) — read + write, indexed format.
- **GLB** (GLTF binary) — read-only, applies node transforms, handles
  GLTF `SCALAR`/`VEC3`/`MAT4` accessors + UNSIGNED_SHORT/INT indices.
- **Format sniffer** (`_formats.sniff_format`) detects magic bytes +
  extension across all supported types.
- **CAD native formats** (`.step`, `.stp`, `.iges`, `.igs`, `.f3d`, `.sldprt`,
  etc.) are detected and cleanly reported as "not a mesh — export from your
  CAD program first" instead of crashing.
- `convert.py` now handles any pair of supported mesh formats, not just
  STL ↔ 3MF.

### New scripts
- **`compat.py`** — identifies which software produced a file and reports a
  compatibility matrix across 15 target programs (Bambu Studio, Orca Slicer,
  PrusaSlicer, Cura, Simplify3D, Fusion 360, SolidWorks, FreeCAD, Blender,
  MeshLab, Tinkercad, OnShape, Printables, MakerWorld, Thingiverse). Flags
  Bambu-specific namespace extensions, Beam Lattice usage, triangle count
  limits, and STL-vs-STEP parametric caveats.
- **`transform.py` no longer silently destroys Bambu Studio / Orca / PrusaSlicer
  files.** v1 only walked the main `3D/3dmodel.model` and missed geometry that
  lived in `3D/Objects/object_N.model` sub-files. Output was a file that looked
  identical to the input, even though the CLI reported success.
- **`convert.py` 3MF→STL produces valid STL files.** v1 wrote a 79-byte header
  instead of the spec-required 80 bytes, misaligning the uint32 triangle count
  field and producing garbage values like `3,774,874,479` that broke round-trips.

### New
- `simplify.py` — vertex-clustering mesh decimation (pure numpy).
- Pure-numpy z-buffer PNG rasterizer (`_raster.py`) — **matplotlib is no longer
  a dependency**. Renders the 225k-tri benchy to 17 KB in ~6s without sampling
  artifacts; the old matplotlib path was 37s with visible needle artifacts.
- Every script now accepts `--json` with a stable output contract (see
  `docs/AGENTS.md`).
- `3dft` single-entry-point dispatcher: `3dft info file.3mf` etc.
- `tests/` — 28 pytest cases, including two named P0-regression tests that
  will fail loudly if either of the above bugs returns.
- GitHub Actions CI on Linux / macOS / Windows × Python 3.9–3.12.
- `docs/AGENTS.md` — integration guide for Claude Code, Claude.ai, Cursor,
  ChatGPT, Manus, Aider, etc.

### Performance
- STL read: ~2s → 0.01s (200× — via vectorized `np.frombuffer`).
- STL write: ~9s → 0.04s (200× — via packed dtype buffer).
- HTML render: per-vertex Python dedup → `np.unique`; base64 Float32Array
  payload instead of per-coord JSON.

### Portability
- `SKILL.md` no longer hard-codes Claude.ai paths. Scripts accept any paths;
  the host agent handles file presentation.
- No matplotlib dependency (was optional before, now removed entirely).
- Stable exit codes (`0`/`1`/`2`) across all scripts.

### Breaking changes
- `render.py` PNG output format changed (new shaded look, 720×720 default).
  Old `--decimate` flag removed — the rasterizer handles any triangle count
  directly.
- `edit_metadata.py` now requires an explicit output path (v1 allowed omitting
  it for `--list`-only).

## v1.0
Initial skill for Claude.ai.
