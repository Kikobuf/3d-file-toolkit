"""
Regression tests for the 3mf-stl-editor scripts.

These tests exercise the CLI surface of every script — the same path that
agents use — and specifically guard against the two P0 bugs that existed in v1:

  1. transform.py silently produced empty output for 3MFs whose geometry
     lived in `3D/Objects/object_N.model` sub-files (the Bambu/Orca pattern).
  2. convert.py wrote a 79-byte STL header, misaligning the triangle-count
     field and corrupting every converted file.

Run:
    pytest tests/          # from the repo root
"""
import os, sys, subprocess, zipfile, struct, json, shutil
import numpy as np
import pytest

HERE       = os.path.dirname(os.path.abspath(__file__))
SCRIPTS    = os.path.abspath(os.path.join(HERE, "..", "scripts"))
FIXTURES   = os.path.join(HERE, "fixtures")
CUBE_STL   = os.path.join(FIXTURES, "cube.stl")
CUBE_FLAT  = os.path.join(FIXTURES, "cube_flat.3mf")
CUBE_EXT   = os.path.join(FIXTURES, "cube_external.3mf")
BENCHY     = os.path.join(FIXTURES, "benchy.3mf")

sys.path.insert(0, SCRIPTS)
from _3mf_reader import read_stl, read_3mf, read_3mf_metadata


# ── helpers ──────────────────────────────────────────────────────────────────

def run(script, *args, expect_ok=True):
    """Invoke a script with --json and parse the summary from stdout."""
    cmd = [sys.executable, os.path.join(SCRIPTS, script), *args, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # 0 = success, 1 = bad args, 2 = operation failed. Anything else is a crash.
    assert proc.returncode in (0, 1, 2), f"{script} crashed: {proc.stderr}"
    # The JSON summary is the last non-empty line
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip().startswith("{")]
    assert lines, f"No JSON in output of {script}: {proc.stdout!r}"
    summary = json.loads(lines[-1])
    if expect_ok:
        assert summary.get("ok") is True, f"{script} failed: {summary}"
    return summary, proc


def bbox_dims(triangles_nx3x3):
    flat = triangles_nx3x3.reshape(-1, 3)
    return (flat.max(0) - flat.min(0))


# ── info.py ──────────────────────────────────────────────────────────────────

def test_info_stl_cube(tmp_path):
    s, _ = run("info.py", CUBE_STL)
    assert s["format"] == "stl"
    assert s["triangles"] == 12
    assert s["vertices_unique"] == 8
    assert np.allclose(s["dimensions_mm"], [10, 10, 10])


def test_info_flat_3mf(tmp_path):
    s, _ = run("info.py", CUBE_FLAT)
    assert s["format"] == "3mf"
    assert len(s["meshes"]) == 1
    assert s["meshes"][0]["triangles"] == 12
    assert np.allclose(s["overall_dimensions_mm"], [10, 10, 10])


def test_info_external_3mf_applies_build_transform():
    """Regression: the Bambu sub-file pattern must apply the build-item translate."""
    s, _ = run("info.py", CUBE_EXT)
    assert s["format"] == "3mf"
    m = s["meshes"][0]
    assert m["triangles"] == 12
    # Build item translates (5,5,0) — bbox should be 5..15 in X/Y, 0..10 in Z
    assert np.allclose(m["bbox_min"], [5, 5, 0])
    assert np.allclose(m["bbox_max"], [15, 15, 10])


# ── transform.py ─────────────────────────────────────────────────────────────

def test_transform_stl_scale(tmp_path):
    out = tmp_path / "cube_2x.stl"
    s, _ = run("transform.py", CUBE_STL, str(out), "--scale", "2.0")
    assert s["triangles"] == 12
    tris = read_stl(str(out))[0]
    assert np.allclose(bbox_dims(tris), [20, 20, 20])


def test_transform_3mf_flat_scale(tmp_path):
    out = tmp_path / "cube_flat_2x.3mf"
    s, _ = run("transform.py", CUBE_FLAT, str(out), "--scale", "2.0")
    assert s["vertices_transformed"] == 8
    assert s["model_files_touched"] == 1
    meshes = read_3mf(str(out))
    assert len(meshes) == 1
    assert np.allclose(bbox_dims(meshes[0]), [20, 20, 20])


def test_transform_3mf_external_scale_P0_REGRESSION(tmp_path):
    """THE P0 regression test.

    v1 silently wrote a 0-triangle output for this input because it only
    walked the main model. If this fails, the critical bug is back.
    """
    out = tmp_path / "cube_ext_2x.3mf"
    s, _ = run("transform.py", CUBE_EXT, str(out), "--scale", "2.0")
    assert s["vertices_transformed"] == 8, "Bambu external sub-file regression: geometry missed"
    meshes = read_3mf(str(out))
    assert len(meshes) == 1, "Output lost its mesh entirely"

    # Build item translates by (5,5,0) (unscaled). Mesh is scaled 2× at model-space,
    # then translated. So output bbox is [5,5,0]..[25,25,20].
    flat = meshes[0].reshape(-1, 3)
    assert np.allclose(flat.min(0), [5, 5, 0]),  f"min wrong: {flat.min(0)}"
    assert np.allclose(flat.max(0), [25, 25, 20]), f"max wrong: {flat.max(0)}"


def test_transform_3mf_translate_only(tmp_path):
    out = tmp_path / "cube_moved.3mf"
    run("transform.py", CUBE_FLAT, str(out), "--translate", "100", "0", "0")
    meshes = read_3mf(str(out))
    flat = meshes[0].reshape(-1, 3)
    assert np.allclose(flat.min(0), [100, 0, 0])
    assert np.allclose(flat.max(0), [110, 10, 10])


def test_transform_3mf_rotate_z_90(tmp_path):
    out = tmp_path / "cube_rot.3mf"
    # Rotate 90° around Z — for a symmetric axis-aligned cube at origin the
    # bbox stays 10x10x10 but shifts to negative X
    run("transform.py", CUBE_FLAT, str(out), "--rotate-z", "90")
    meshes = read_3mf(str(out))
    dims = bbox_dims(meshes[0])
    assert np.allclose(dims, [10, 10, 10], atol=1e-4)


def test_transform_rejects_cross_format(tmp_path):
    out = tmp_path / "cube.stl"
    s, _ = run("transform.py", CUBE_FLAT, str(out), "--scale", "2.0",
               expect_ok=False)
    assert s["ok"] is False
    assert "extensions differ" in s["error"].lower()


# ── convert.py ───────────────────────────────────────────────────────────────

def test_convert_stl_to_3mf(tmp_path):
    out = tmp_path / "cube.3mf"
    s, _ = run("convert.py", CUBE_STL, str(out))
    assert s["triangles"] == 12
    meshes = read_3mf(str(out))
    assert len(meshes) == 1
    assert np.allclose(bbox_dims(meshes[0]), [10, 10, 10])


def test_convert_3mf_to_stl_header_is_80_bytes_P0_REGRESSION(tmp_path):
    """THE other P0 regression test.

    v1 wrote a 79-byte header, misaligning the uint32 triangle count field
    by one byte and producing garbage counts like 3_774_874_479.
    """
    out = tmp_path / "cube.stl"
    s, _ = run("convert.py", CUBE_FLAT, str(out))
    assert s["triangles"] == 12

    # Spot-check the file bytes directly
    with open(out, "rb") as f:
        header = f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
    assert len(header) == 80, "STL header must be exactly 80 bytes"
    assert count == 12, f"Header tri count wrong: {count}"
    # File size must match:  84 + 50 * count
    assert os.path.getsize(out) == 84 + 50 * 12


def test_convert_3mf_external_to_stl_applies_build_transform(tmp_path):
    """The STL produced from a Bambu-style 3MF should be in world space."""
    out = tmp_path / "ext.stl"
    run("convert.py", CUBE_EXT, str(out))
    tris = read_stl(str(out))[0]
    flat = tris.reshape(-1, 3)
    # Bbox should reflect the (5,5,0) build-item translate
    assert np.allclose(flat.min(0), [5, 5, 0])
    assert np.allclose(flat.max(0), [15, 15, 10])


# ── merge.py ─────────────────────────────────────────────────────────────────

def test_merge_two_cubes_to_stl(tmp_path):
    out = tmp_path / "merged.stl"
    s, _ = run("merge.py", CUBE_STL, CUBE_FLAT, "-o", str(out))
    assert s["total_meshes"] == 2
    assert s["total_triangles"] == 24
    tris = read_stl(str(out))[0]
    assert len(tris) == 24


def test_merge_two_cubes_to_3mf(tmp_path):
    out = tmp_path / "merged.3mf"
    s, _ = run("merge.py", CUBE_STL, CUBE_FLAT, "-o", str(out))
    assert s["total_meshes"] == 2
    meshes = read_3mf(str(out))
    assert len(meshes) == 2


def test_merge_includes_external_3mf(tmp_path):
    """Merging a Bambu-style 3MF must preserve its transformed geometry."""
    out = tmp_path / "merged.stl"
    s, _ = run("merge.py", CUBE_EXT, "-o", str(out))
    assert s["total_triangles"] == 12
    tris = read_stl(str(out))[0]
    flat = tris.reshape(-1, 3)
    assert np.allclose(flat.min(0), [5, 5, 0])


# ── split.py ─────────────────────────────────────────────────────────────────

def test_split_list(tmp_path):
    s, _ = run("split.py", CUBE_EXT, "--list")
    assert len(s["meshes"]) == 1
    assert s["meshes"][0]["triangles"] == 12


def test_split_extract_external(tmp_path):
    out = tmp_path / "extracted.stl"
    s, _ = run("split.py", CUBE_EXT, "-o", str(out), "--object-index", "0")
    assert s["triangles"] == 12
    tris = read_stl(str(out))[0]
    assert len(tris) == 12


# ── edit_metadata.py ─────────────────────────────────────────────────────────

def test_metadata_list(tmp_path):
    s, _ = run("edit_metadata.py", CUBE_FLAT, "--list")
    assert s["metadata"].get("Title") == "Test Cube (flat)"


def test_metadata_set(tmp_path):
    out = tmp_path / "tagged.3mf"
    s, _ = run("edit_metadata.py", CUBE_FLAT, str(out),
               "--set", "Designer=Enrico",
               "--set", "Title=KB Rides Test Cube")
    assert "Designer" in s["added"]
    assert "Title" in s["set"]
    meta = read_3mf_metadata(str(out))
    assert meta["Designer"] == "Enrico"
    assert meta["Title"] == "KB Rides Test Cube"


def test_metadata_remove(tmp_path):
    out = tmp_path / "scrubbed.3mf"
    s, _ = run("edit_metadata.py", CUBE_FLAT, str(out), "--remove", "Title")
    assert "Title" in s["removed"]
    meta = read_3mf_metadata(str(out))
    assert "Title" not in meta


# ── simplify.py ──────────────────────────────────────────────────────────────

def test_simplify_stl_by_ratio(tmp_path):
    """Simplifying a small cube by ratio should still be a cube-ish shape."""
    out = tmp_path / "cube_simp.stl"
    s, _ = run("simplify.py", CUBE_STL, str(out), "--ratio", "0.9")
    assert s["total_input_tris"] == 12
    # Cubes are already minimal — simplifier should return equal or fewer tris
    assert s["total_output_tris"] <= 12
    tris = read_stl(str(out))[0]
    # Bbox should still be roughly 10x10x10
    dims = bbox_dims(tris)
    assert np.allclose(dims, [10, 10, 10], atol=0.5)


def test_simplify_3mf_to_target_count(tmp_path):
    """Heavier simplification should not grow the triangle count."""
    out = tmp_path / "cube_simp.3mf"
    # A 10mm grid on a 10mm cube collapses opposite-corner verts, killing the mesh
    s, _ = run("simplify.py", CUBE_FLAT, str(out), "--grid", "20.0")
    assert s["total_output_tris"] <= s["total_input_tris"]
    assert s["ok"] is True


def test_simplify_preserves_format(tmp_path):
    """Output extension dictates format — don't cross extensions."""
    out_stl = tmp_path / "a.stl"
    out_3mf = tmp_path / "b.3mf"
    run("simplify.py", CUBE_STL, str(out_stl), "--ratio", "0.5")
    run("simplify.py", CUBE_FLAT, str(out_3mf), "--ratio", "0.5")
    assert out_stl.exists() and out_3mf.exists()


@pytest.mark.skipif(not os.path.exists(BENCHY), reason="benchy fixture not present")
def test_simplify_benchy_to_max_tris(tmp_path):
    out = tmp_path / "benchy_small.3mf"
    s, _ = run("simplify.py", BENCHY, str(out), "--max-tris", "10000")
    assert s["total_output_tris"] <= 15000, f"Simplifier overshot: {s['total_output_tris']}"
    # Sanity — rough bbox must survive
    info, _ = run("info.py", str(out))
    dims = info["overall_dimensions_mm"]
    assert 55 < dims[0] < 65
    assert 28 < dims[1] < 34
    assert 45 < dims[2] < 51


# ── render.py ────────────────────────────────────────────────────────────────

def test_render_html_produces_valid_file(tmp_path):
    out = tmp_path / "cube.html"
    s, _ = run("render.py", CUBE_FLAT, str(out))
    assert out.exists()
    html = out.read_text()
    # Sanity: file contains base64 binary blob marker and mesh metadata
    assert "BLOB_B64" in html
    assert 'canvas id="c"' in html
    assert "drawElements" in html


def test_render_html_external_3mf_not_empty(tmp_path):
    """Regression — render HTML for Bambu-external must actually include geometry."""
    out = tmp_path / "ext.html"
    s, _ = run("render.py", CUBE_EXT, str(out))
    assert s["triangles"] == 12
    # Loose sanity — HTML for a non-empty cube should be a few KB at least
    assert os.path.getsize(out) > 1000


# ── optional benchy integration test ─────────────────────────────────────────
#   Only runs if the benchy fixture is present (it's ~3 MB, not shipped in git).

@pytest.mark.skipif(not os.path.exists(BENCHY), reason="benchy fixture not present")
def test_benchy_info():
    s, _ = run("info.py", BENCHY)
    assert s["format"] == "3mf"
    assert s["meshes"][0]["triangles"] == 225154
    assert np.allclose(s["overall_dimensions_mm"], [60.0, 31.0, 48.0], atol=0.01)


@pytest.mark.skipif(not os.path.exists(BENCHY), reason="benchy fixture not present")
def test_benchy_scale_roundtrip(tmp_path):
    out = tmp_path / "benchy_half.3mf"
    s, _ = run("transform.py", BENCHY, str(out), "--scale", "0.5")
    assert s["vertices_transformed"] > 100_000
    s2, _ = run("info.py", str(out))
    assert np.allclose(s2["overall_dimensions_mm"], [30.0, 15.5, 24.0], atol=0.05)


# ── multi-format support (OBJ, PLY, GLB, format sniffing) ──────────────────

def test_convert_3mf_to_obj(tmp_path):
    out = tmp_path / "cube.obj"
    s, _ = run("convert.py", CUBE_FLAT, str(out))
    assert s["format_out"] == "obj"
    assert out.exists() and out.stat().st_size > 0
    # Must re-load the OBJ and get the same triangle count
    from _formats import read_obj
    m = read_obj(str(out))
    assert sum(len(x) for x in m) == 12


def test_convert_stl_to_ply_roundtrip(tmp_path):
    """STL -> PLY -> STL should preserve triangle count and bounding box."""
    ply_path = tmp_path / "cube.ply"
    stl_path = tmp_path / "cube2.stl"
    run("convert.py", CUBE_STL, str(ply_path))
    run("convert.py", str(ply_path), str(stl_path))
    t = read_stl(str(stl_path))[0]
    assert len(t) == 12
    assert np.allclose(bbox_dims(t), [10, 10, 10])


def test_convert_obj_to_3mf_roundtrip(tmp_path):
    """Write an OBJ by hand, convert it to 3MF, verify bbox/tri count."""
    from _formats import write_obj
    from _3mf_reader import read_stl
    obj_path = tmp_path / "cube.obj"
    out_3mf  = tmp_path / "cube.3mf"
    tris = read_stl(CUBE_STL)[0]
    write_obj(str(obj_path), [tris])
    run("convert.py", str(obj_path), str(out_3mf))
    meshes = read_3mf(str(out_3mf))
    assert sum(len(m) for m in meshes) == 12
    assert np.allclose(bbox_dims(meshes[0]), [10, 10, 10])


def test_sniff_format_detects_kinds(tmp_path):
    """Run the detector directly — no subprocess."""
    from _formats import sniff_format
    assert sniff_format(CUBE_STL)["kind"] == "mesh"
    assert sniff_format(CUBE_FLAT)["kind"] == "mesh"
    # Fake CAD file — just need the extension
    fake_step = tmp_path / "fake.step"
    fake_step.write_bytes(b"ISO-10303-21;\nHEADER;\n")
    assert sniff_format(str(fake_step))["kind"] == "cad"
    # Unknown
    fake = tmp_path / "fake.xyz"
    fake.write_bytes(b"junk")
    assert sniff_format(str(fake))["kind"] == "unknown"


def test_read_rejects_cad_format(tmp_path):
    """The uniform read() function should raise for STEP files."""
    from _formats import read
    fake = tmp_path / "fake.stp"
    fake.write_bytes(b"ISO-10303-21;\n")
    with pytest.raises(ValueError, match="CAD"):
        read(str(fake))


# ── compat.py ──────────────────────────────────────────────────────────────

def test_compat_detects_bambu_on_benchy():
    if not os.path.exists(BENCHY):
        pytest.skip("benchy fixture not present")
    s, _ = run("compat.py", BENCHY)
    assert "Bambu" in s["detected"]["writer"]
    assert "bambu_studio_namespace" in s["detected"]["extensions"]


def test_compat_stl_unknown_writer():
    s, _ = run("compat.py", CUBE_STL)
    # Our own fixture was generated by this tool, so detection should catch it
    assert "3mf-stl-editor" in s["detected"]["writer"] or s["detected"]["writer"] == "unknown"


def test_compat_single_target():
    if not os.path.exists(BENCHY):
        pytest.skip("benchy fixture not present")
    s, _ = run("compat.py", BENCHY, "--target", "tinkercad")
    assert "tinkercad" in s["compat"]
    assert s["compat"]["tinkercad"]["verdict"] == "blocked"


def test_compat_cad_file_detected(tmp_path):
    fake = tmp_path / "fake.step"
    fake.write_bytes(b"ISO-10303-21;\nHEADER;\n")
    s, _ = run("compat.py", str(fake))
    assert s["kind"] == "cad"


# ── GLB write (new in v2.1) ────────────────────────────────────────────────

def test_convert_3mf_to_glb_and_back(tmp_path):
    """3MF -> GLB -> STL round-trip: triangle count and bbox must survive."""
    glb_path = tmp_path / "cube.glb"
    stl_path = tmp_path / "cube_rt.stl"
    run("convert.py", CUBE_FLAT, str(glb_path))
    assert glb_path.exists() and glb_path.stat().st_size > 0

    # Verify it's a real GLB (magic bytes "glTF")
    with open(glb_path, "rb") as f:
        magic = f.read(4)
    assert magic == b"glTF", f"Bad GLB magic: {magic!r}"

    # Round-trip through STL
    run("convert.py", str(glb_path), str(stl_path))
    tris = read_stl(str(stl_path))[0]
    assert len(tris) == 12
    assert np.allclose(bbox_dims(tris), [10, 10, 10])


def test_glb_write_multi_mesh(tmp_path):
    """Merging two cubes to GLB should produce a valid file with 24 tris."""
    merged_3mf = tmp_path / "merged.3mf"
    glb_path   = tmp_path / "merged.glb"
    run("merge.py", CUBE_STL, CUBE_FLAT, "-o", str(merged_3mf))
    run("convert.py", str(merged_3mf), str(glb_path))
    with open(glb_path, "rb") as f:
        assert f.read(4) == b"glTF"


# ── batch metadata (new in v2.1) ──────────────────────────────────────────

def test_metadata_batch(tmp_path):
    """--batch should update all matching 3MF files in-place."""
    import shutil
    a = tmp_path / "a.3mf"; b = tmp_path / "b.3mf"
    shutil.copy(CUBE_FLAT, a); shutil.copy(CUBE_FLAT, b)

    s, _ = run("edit_metadata.py", str(tmp_path / "*.3mf"),
               "--batch", "--set", "Designer=BatchTest")
    assert s["processed"] == 2

    from _3mf_reader import read_3mf_metadata
    assert read_3mf_metadata(str(a)).get("Designer") == "BatchTest"
    assert read_3mf_metadata(str(b)).get("Designer") == "BatchTest"


def test_metadata_template(tmp_path):
    """--template should load KEY/VALUE pairs from a JSON file."""
    import json as _json
    tmpl = tmp_path / "meta.json"
    tmpl.write_text(_json.dumps({"Designer": "TemplateUser", "License": "MIT"}))
    out = tmp_path / "templated.3mf"
    s, _ = run("edit_metadata.py", CUBE_FLAT, str(out), "--template", str(tmpl))
    assert "Designer" in s["added"] or "Designer" in s["set"]
    from _3mf_reader import read_3mf_metadata
    meta = read_3mf_metadata(str(out))
    assert meta["Designer"] == "TemplateUser"
    assert meta["License"] == "MIT"
