# Integrating with different AI agents

The scripts in this skill are just Python CLIs — they work anywhere Python
does. This doc covers how to wire them up for each major agent so the agent
reliably finds and invokes them when the user drops a 3D file.

The general principles:

1. Tell the agent where `scripts/` lives.
2. Tell it to prefer `--json` when parsing results.
3. Tell it to call `transform.py` / `convert.py` / etc. with absolute paths.
4. Let it handle showing files to the user through whatever file-presentation
   mechanism the host offers. Don't hard-code `/mnt/...` anywhere.

---

## Claude Code (CLI)

Drop this repo into `~/.claude/skills/3d-file-toolkit/`:

```bash
git clone https://github.com/<you>/3d-file-toolkit.git ~/.claude/skills/3d-file-toolkit
pip install -r ~/.claude/skills/3d-file-toolkit/requirements.txt
```

Claude Code auto-discovers `SKILL.md` files in `~/.claude/skills/`. The skill
description triggers on any `.stl` / `.3mf` file mention, and the scripts run
in the user's active shell so they can read files from anywhere in the project.

**Test**: `claude "scale the benchy.3mf in my downloads folder by 2x"`.

## Claude.ai (web / desktop)

Claude.ai mounts skills at `/mnt/skills/user/<n>/`. To use this skill
there, upload the folder through the skill manager or paste the scripts
directly. Inside that sandbox:

- Uploads land at `/mnt/user-data/uploads/`.
- Outputs should be copied to `/mnt/user-data/outputs/` and announced with
  Claude's `present_files` tool.
- The scripts themselves don't care about those paths — just pass them as
  positional arguments.

## Cursor / Continue / Cline (VS Code agents)

These agents don't have a formal "skill" concept. The cleanest integration
is a `.cursor/rules/3d-file-toolkit.md` (or `.continue/rules/` etc.) with the
trigger text from `SKILL.md` plus the absolute path to the scripts:

```md
# 3MF/STL file handling

Whenever the user mentions a `.3mf` or `.stl` file, use the scripts in
`/path/to/3d-file-toolkit/scripts/` before writing any new code.

Prefer `--json` output for programmatic parsing.

Available:
- info.py   — stats + metadata
- render.py — .html (WebGL) or .png preview
- transform.py — scale/translate/rotate
- convert.py — .stl ↔ .3mf
- merge.py / split.py / edit_metadata.py
```

Keep the repo checked out anywhere on the dev machine; the agent will call
`python3 /abs/path/scripts/info.py file.3mf` directly.

## ChatGPT Code Interpreter / Custom GPT with Code

Upload the `scripts/` folder to the session (or include it in a Custom GPT's
knowledge). Instruct the GPT in the system prompt:

> When the user uploads a .stl or .3mf file, invoke the scripts in
> `/mnt/data/scripts/` using subprocess. Always pass `--json` and parse the
> output rather than reading printed text.

ChatGPT's sandbox has numpy preinstalled, which is the only dependency —
PNG rendering uses an embedded pure-numpy rasterizer, not matplotlib.

## Manus

Manus runs long-horizon shell tasks in its own VM. Install the skill once:

```bash
git clone https://github.com/<you>/3d-file-toolkit.git /opt/3d-file-toolkit
pip install -r /opt/3d-file-toolkit/requirements.txt
```

Then in the Manus task prompt:

> Use `/opt/3d-file-toolkit/scripts/*.py` for anything involving .stl or .3mf
> files. Always pass `--json`.

## Aider / Goose / other open-source agents

Point the repo at any path and include its `README.md` in the agent's context.
These agents generally reason well from docs alone — no special integration
needed beyond making sure numpy is installed in their Python.

---

## JSON output contract (stable across versions)

All scripts emit a single-line JSON object on their last line of stdout when
`--json` is passed. Minimal shape:

```json
{"ok": true, "output": "/path/to/output.3mf", ...}
```

`ok: false` is accompanied by `"error": "..."` and a non-zero exit code.

Per-script fields (additive — never removed):

| Script             | Extra fields                                             |
|--------------------|----------------------------------------------------------|
| `info.py`          | `format`, `meshes[]`, `overall_dimensions_mm`, `metadata`, `zip_entries` |
| `transform.py`     | `vertices_transformed`, `model_files_touched`, `new_dimensions_mm` |
| `convert.py`       | `format_in`, `format_out`, `triangles`, `input_meshes`   |
| `merge.py`         | `input_files[]`, `total_meshes`, `total_triangles`       |
| `split.py`         | `index`, `name`, `triangles` (extract) · `meshes[]` (--list) |
| `edit_metadata.py` | `added[]`, `set[]`, `removed[]`, `missing[]`             |
| `simplify.py`      | `grid_mm`, `total_input_tris`, `total_output_tris`, `reduction` |
| `render.py`        | `format` (html/png), `size_bytes`, `renderer`            |
| `compat.py`        | `detected.writer`, `detected.extensions`, `health`, `compat{target: {verdict, notes}}` |

Exit codes:

- `0` — success.
- `1` — bad arguments, missing input, unsupported format.
- `2` — operation failed (e.g. 3MF had no mesh data).

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'numpy'"**
Run `pip install numpy`. It's the only runtime dependency.

**Transform succeeded but output file looks unchanged in slicer**
Re-open in the slicer with a fresh import (some slicers cache the old
geometry). If the JSON summary reports `vertices_transformed: 0`, the input
is a malformed or signed 3MF — re-export from the source.

**"3MF had no mesh data" on a file that clearly has geometry**
Usually means the input is from a slicer that stores geometry in a format
variant this tool doesn't yet support (e.g. compressed meshes, or
the 3MF Beam Lattice Extension). File an issue with the problematic file.
