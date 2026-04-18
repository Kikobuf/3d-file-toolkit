"""
Shared 3MF/STL I/O module.

3MF structure:
  <build>
    <item objectid="X" transform="..."/>       # place object on build plate
  </build>
  <resources>
    <object id="X">                            # logical object
      <components>                             # OR: references to sub-objects
        <component objectid="Y" transform="..." p:path="/3D/Objects/Y.model"/>
      </components>
    </object>
    <object id="Y"><mesh>...</mesh></object>   # OR: direct mesh
  </resources>

Transforms are 3x4 column-major 4x4 matrices (last row implied as [0 0 0 1]).
Resolution order for a mesh: item_transform @ component_transform @ ... @ mesh

Bambu Studio, Orca, PrusaSlicer put the actual geometry in separate
`3D/Objects/object_N.model` sub-files and reference them from the main
`3D/3dmodel.model` via `<component p:path=".../object_N.model"/>`.
This module walks those references and applies the full transform chain.
"""
import zipfile, struct, os, re
import xml.etree.ElementTree as ET
import numpy as np

NS   = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
P_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
tag  = lambda t: f"{{{NS}}}{t}"
ptag = lambda t: f"{{{P_NS}}}{t}"


# ──────────────────────────────────────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────────────────────────────────────

def parse_transform(s):
    """Parse a 3MF 3x4 column-major transform string into a 4x4 numpy matrix."""
    if not s or not s.strip():
        return np.eye(4, dtype=np.float64)
    v = [float(x) for x in s.split()]
    if len(v) < 12:
        return np.eye(4, dtype=np.float64)
    m = np.eye(4, dtype=np.float64)
    m[0,0]=v[0];  m[1,0]=v[1];  m[2,0]=v[2]
    m[0,1]=v[3];  m[1,1]=v[4];  m[2,1]=v[5]
    m[0,2]=v[6];  m[1,2]=v[7];  m[2,2]=v[8]
    m[0,3]=v[9];  m[1,3]=v[10]; m[2,3]=v[11]
    return m


def format_transform(mat4):
    """Serialize a 4x4 matrix back to a 3MF 3x4 column-major transform string."""
    v = [
        mat4[0,0], mat4[1,0], mat4[2,0],
        mat4[0,1], mat4[1,1], mat4[2,1],
        mat4[0,2], mat4[1,2], mat4[2,2],
        mat4[0,3], mat4[1,3], mat4[2,3],
    ]
    return " ".join(f"{x:.6f}" for x in v)


def apply_transform(verts_nx3, mat4):
    """Apply a 4x4 matrix to (N,3) vertices."""
    if np.allclose(mat4, np.eye(4)):
        return verts_nx3.astype(np.float32)
    ones = np.ones((len(verts_nx3), 1), dtype=np.float64)
    v4 = np.hstack([verts_nx3.astype(np.float64), ones])
    return (mat4 @ v4.T).T[:, :3].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Fast bulk XML parsing
# ──────────────────────────────────────────────────────────────────────────────
#  ElementTree's findall + .get() per-element is O(N) Python; on 200k verts it
#  takes seconds. We short-circuit by regex-scanning the raw XML bytes for
#  vertex/triangle attributes — 10-50× faster, equivalent correctness for
#  well-formed 3MF output.

_VERT_RE = re.compile(rb'<(?:\w+:)?vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"')
_TRI_RE  = re.compile(rb'<(?:\w+:)?triangle\s+v1="([^"]+)"\s+v2="([^"]+)"\s+v3="([^"]+)"')

def _parse_mesh_fast(mesh_xml_bytes):
    """Regex-parse <vertex>/<triangle> from a mesh XML blob. Returns (verts, tris) or (None, None)."""
    vm = _VERT_RE.findall(mesh_xml_bytes)
    tm = _TRI_RE.findall(mesh_xml_bytes)
    if not vm or not tm:
        return None, None
    verts = np.array(vm, dtype=np.float32)
    tris  = np.array(tm, dtype=np.int32)
    return verts, tris


def _parse_mesh_element(mesh_el):
    """ElementTree fallback parser. Slower, handles odd attribute ordering."""
    ve = mesh_el.find(tag("vertices"))
    te = mesh_el.find(tag("triangles"))
    if ve is None or te is None:
        return None, None
    verts = np.array(
        [[float(v.get("x",0)), float(v.get("y",0)), float(v.get("z",0))]
         for v in ve.findall(tag("vertex"))], dtype=np.float32)
    tris = np.array(
        [[int(t.get("v1",0)), int(t.get("v2",0)), int(t.get("v3",0))]
         for t in te.findall(tag("triangle"))], dtype=np.int32)
    return verts, tris


# ──────────────────────────────────────────────────────────────────────────────
#  3MF reading (transform-aware)
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_file(zf, path):
    """Load a .model file from the ZIP, return (root_element, raw_bytes)."""
    with zf.open(path) as f:
        raw = f.read()
    root = ET.fromstring(raw)
    return root, raw


def _find_obj_mesh_bytes(raw_bytes, obj_id):
    """Slice raw model bytes to isolate <object id="X">...</object> contents.
    Used for the fast regex parser, so we don't cross object boundaries."""
    # Loose search — works for typical slicer output. Falls back to whole doc if not found.
    pattern = re.compile(
        rb'<(?:\w+:)?object\s+[^>]*\bid="' + re.escape(obj_id.encode()) + rb'"[^>]*>(.*?)</(?:\w+:)?object>',
        re.DOTALL)
    m = pattern.search(raw_bytes)
    return m.group(1) if m else raw_bytes


def _resolve_object(obj_el, obj_map, zf, xform, meshes_out, raw_bytes_by_path, current_path):
    """Recursively resolve an object, collecting (name, world-space triangle soup) tuples."""
    name = obj_el.get("name") or f"object_{obj_el.get('id','?')}"

    mesh_el = obj_el.find(tag("mesh"))
    if mesh_el is not None:
        # Try fast path first — slice the raw bytes down to this object's region
        obj_id = obj_el.get("id", "")
        raw_slice = _find_obj_mesh_bytes(raw_bytes_by_path.get(current_path, b""), obj_id)
        verts, tri_idx = _parse_mesh_fast(raw_slice)
        if verts is None:
            verts, tri_idx = _parse_mesh_element(mesh_el)
        if verts is not None and len(verts) and len(tri_idx):
            verts_world = apply_transform(verts, xform)
            meshes_out.append((name, verts_world[tri_idx]))

    comps_el = obj_el.find(tag("components"))
    if comps_el is None:
        return
    for comp in comps_el.findall(tag("component")):
        comp_xform = parse_transform(comp.get("transform", ""))
        combined = xform @ comp_xform
        ext_path = comp.get(ptag("path"), "").lstrip("/")
        comp_obj_id = comp.get("objectid", "")

        if ext_path and zf is not None and ext_path in zf.namelist():
            if ext_path not in raw_bytes_by_path:
                ext_root, ext_raw = _load_model_file(zf, ext_path)
                raw_bytes_by_path[ext_path] = ext_raw
            else:
                ext_root = ET.fromstring(raw_bytes_by_path[ext_path])
            ext_res = ext_root.find(tag("resources"))
            if ext_res is not None:
                ext_obj_map = {o.get("id"): o for o in ext_res.findall(tag("object"))}
                target = ext_obj_map.get(comp_obj_id)
                if target is None and len(ext_obj_map) > 0:
                    target = next(iter(ext_obj_map.values()))
                if target is not None:
                    _resolve_object(target, ext_obj_map, zf, combined, meshes_out,
                                    raw_bytes_by_path, ext_path)
        elif comp_obj_id in obj_map:
            _resolve_object(obj_map[comp_obj_id], obj_map, zf, combined, meshes_out,
                            raw_bytes_by_path, current_path)


def read_3mf(path, with_names=False):
    """Read all meshes from a 3MF with full transform chain applied.
    Returns list of world-space (N,3,3) triangle arrays (one per logical mesh).
    If with_names=True, returns list of (name, tris) tuples."""
    results = []
    with zipfile.ZipFile(path, "r") as zf:
        main_model = next((n for n in zf.namelist() if n.endswith("3dmodel.model")), None)
        if not main_model:
            return results
        root, raw = _load_model_file(zf, main_model)
        raw_bytes_by_path = {main_model: raw}

        resources = root.find(tag("resources"))
        if resources is None:
            return results
        obj_map = {o.get("id"): o for o in resources.findall(tag("object"))}

        build = root.find(tag("build"))
        build_items = build.findall(tag("item")) if build is not None else []

        meshes_out = []
        if build_items:
            for item in build_items:
                item_xform = parse_transform(item.get("transform", ""))
                obj_id = item.get("objectid", "")
                if obj_id in obj_map:
                    _resolve_object(obj_map[obj_id], obj_map, zf, item_xform,
                                    meshes_out, raw_bytes_by_path, main_model)
        else:
            for obj in obj_map.values():
                _resolve_object(obj, obj_map, zf, np.eye(4),
                                meshes_out, raw_bytes_by_path, main_model)

        results = meshes_out
    return results if with_names else [m for _, m in results]


def read_3mf_metadata(path):
    """Return dict of 3MF metadata from the main model file."""
    with zipfile.ZipFile(path, "r") as zf:
        main_model = next((n for n in zf.namelist() if n.endswith("3dmodel.model")), None)
        if not main_model:
            return {}
        with zf.open(main_model) as f:
            root = ET.parse(f).getroot()
    return {m.get("name",""): (m.text or "") for m in root.findall(tag("metadata"))}


# ──────────────────────────────────────────────────────────────────────────────
#  3MF structure inspection (for transform-in-place edits)
# ──────────────────────────────────────────────────────────────────────────────

def list_mesh_sources(path):
    """List every (model_file, object_id) tuple that contains a <mesh> in the ZIP.
    Used by transform.py to know which files to rewrite."""
    sources = []
    with zipfile.ZipFile(path, "r") as zf:
        for mf in zf.namelist():
            if not mf.endswith(".model"):
                continue
            with zf.open(mf) as f:
                root = ET.parse(f).getroot()
            resources = root.find(tag("resources"))
            if resources is None:
                continue
            for obj in resources.findall(tag("object")):
                if obj.find(tag("mesh")) is not None:
                    sources.append((mf, obj.get("id", "")))
    return sources


# ──────────────────────────────────────────────────────────────────────────────
#  STL
# ──────────────────────────────────────────────────────────────────────────────

def read_stl_binary(path):
    with open(path, "rb") as f:
        f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
        # Bulk read all triangles at once
        data = f.read(count * 50)
    if len(data) < count * 50:
        count = len(data) // 50
    # Each triangle: 12-byte normal + 3×12-byte verts + 2-byte attr = 50 bytes
    dt = np.dtype([
        ("n", "<f4", 3),
        ("v", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    arr = np.frombuffer(data[:count*50], dtype=dt)
    return [arr["v"].astype(np.float32)]


def read_stl_ascii(path):
    tris, cur = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            l = line.strip()
            if l.startswith("vertex"):
                p = l.split()
                cur.append([float(p[1]), float(p[2]), float(p[3])])
            elif l.startswith("endfacet"):
                if len(cur) == 3: tris.append(cur)
                cur = []
    return [np.array(tris, dtype=np.float32)]


def read_stl(path):
    with open(path, "rb") as f:
        first = f.read(6)
    # ASCII STL starts with literal "solid " followed by human-readable text.
    # Binary STL has an 80-byte arbitrary header — some binary writers also start with "solid",
    # so we have to sanity-check: if the declared triangle count matches file size, it's binary.
    if first.startswith(b"solid"):
        sz = os.path.getsize(path)
        with open(path, "rb") as f:
            f.read(80)
            try:
                count = struct.unpack("<I", f.read(4))[0]
                expected = 84 + count * 50
                if expected == sz:
                    return read_stl_binary(path)
            except Exception:
                pass
        return read_stl_ascii(path)
    return read_stl_binary(path)


def write_stl_binary(path, tris, header_text="3mf-stl-editor"):
    """Write triangles to a binary STL.
    tris: (N, 3, 3) numpy array of N triangles × 3 verts × 3 coords.
    Header is exactly 80 bytes, followed by uint32 triangle count.
    """
    tris = np.asarray(tris, dtype=np.float32)
    n = len(tris)
    header = header_text.encode("ascii", errors="replace")[:80]
    header = header.ljust(80, b" ")                           # <-- exactly 80 bytes
    assert len(header) == 80, f"STL header is {len(header)} bytes, must be 80"

    # Vectorized normal computation
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = (normals / lengths).astype(np.float32)

    # Build the packed buffer in one go
    dt = np.dtype([
        ("n", "<f4", 3),
        ("v", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    buf = np.zeros(n, dtype=dt)
    buf["n"] = normals
    buf["v"] = tris
    buf["attr"] = 0

    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n))
        f.write(buf.tobytes())


# ──────────────────────────────────────────────────────────────────────────────
#  Dispatch
# ──────────────────────────────────────────────────────────────────────────────

def load_meshes(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl": return read_stl(path)
    if ext == ".3mf": return read_3mf(path)
    raise ValueError(f"Unknown extension '{ext}'")
