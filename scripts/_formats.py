"""
Multi-format 3D file I/O.

Readers return list of (N, 3, 3) float32 triangle arrays — world-space triangle
soup, same contract as `_3mf_reader.read_3mf`. Writers take the same shape.

Natively supported (mesh formats, pure numpy + stdlib):
    .stl   — binary + ASCII (handled in _3mf_reader)
    .3mf   — ZIP + XML (handled in _3mf_reader)
    .obj   — Wavefront ASCII; mesh geometry only (materials ignored)
    .ply   — binary little-endian + ASCII; geometry only
    .glb   — GLTF binary (read-only in this version)

Detected but NOT editable (CAD native formats — require a CAD kernel):
    .step / .stp / .iges / .igs / .f3d / .f3z / .stl.gz

Call `sniff_format(path)` to identify any supported file. Call `read(path)`
for a uniform triangle-soup load, or use the format-specific helpers directly.
"""
import os, struct, json, zipfile, re
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Format detection (magic bytes + extension)
# ──────────────────────────────────────────────────────────────────────────────

MESH_EXTS = {".stl", ".3mf", ".obj", ".ply", ".glb", ".gltf"}
CAD_EXTS  = {".step", ".stp", ".iges", ".igs", ".f3d", ".f3z", ".sldprt", ".ipt"}

def sniff_format(path):
    """Return a dict describing the file: {ext, kind, detail}.
    kind is one of: 'mesh', 'cad', 'archive', 'unknown'."""
    ext = os.path.splitext(path)[1].lower()
    size = os.path.getsize(path) if os.path.exists(path) else 0

    info = {"path": path, "ext": ext, "size_bytes": size}

    if ext not in MESH_EXTS and ext not in CAD_EXTS:
        info["kind"] = "unknown"
        info["detail"] = f"Unrecognized extension {ext!r}"
        return info

    if ext in CAD_EXTS:
        info["kind"] = "cad"
        info["detail"] = {
            ".step": "STEP (ISO 10303) — CAD B-rep, requires CAD kernel to mesh",
            ".stp":  "STEP (ISO 10303) — CAD B-rep, requires CAD kernel to mesh",
            ".iges": "IGES — legacy CAD B-rep, requires CAD kernel",
            ".igs":  "IGES — legacy CAD B-rep, requires CAD kernel",
            ".f3d":  "Fusion 360 native file",
            ".f3z":  "Fusion 360 archive",
            ".sldprt": "SolidWorks Part — proprietary",
            ".ipt":  "Inventor Part — proprietary",
        }.get(ext, "CAD native format")
        return info

    # Mesh formats — peek magic bytes for a bit more detail
    with open(path, "rb") as f:
        head = f.read(16)
    info["kind"] = "mesh"
    if ext == ".stl":
        info["detail"] = "STL ASCII" if head.startswith(b"solid") and _looks_ascii_stl(path) else "STL binary"
    elif ext == ".3mf":
        info["detail"] = "3MF (ZIP container with XML model)"
    elif ext == ".obj":
        info["detail"] = "Wavefront OBJ (ASCII)"
    elif ext == ".ply":
        with open(path, "rb") as f:
            line = f.readline()
            fmt_line = f.readline()
        fmt = "?"
        if b"format binary_little_endian" in fmt_line: fmt = "binary LE"
        elif b"format binary_big_endian" in fmt_line:  fmt = "binary BE"
        elif b"format ascii" in fmt_line:              fmt = "ASCII"
        info["detail"] = f"PLY ({fmt})"
    elif ext == ".glb":
        info["detail"] = "GLTF binary (.glb)" if head.startswith(b"glTF") else "GLB (wrong magic)"
    elif ext == ".gltf":
        info["detail"] = "GLTF JSON"
    return info


def _looks_ascii_stl(path):
    """ASCII STL files have a triangle count that doesn't match file size,
    unlike binary STLs that happen to start with 'solid'."""
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as f:
            f.read(80)
            count = struct.unpack("<I", f.read(4))[0]
        return sz != 84 + count * 50
    except Exception:
        return True


# ──────────────────────────────────────────────────────────────────────────────
#  Wavefront OBJ
# ──────────────────────────────────────────────────────────────────────────────

def read_obj(path):
    """Read an OBJ file. Ignores materials, textures, groups; triangulates
    n-gons via fan triangulation. Returns list of (N,3,3) float32 arrays —
    one entry per `o` or `g` grouping, or a single entry if none exist."""
    verts = []
    groups = []            # list of (name, [tri_indices])
    current_name = "default"
    current_tris = []

    def _flush():
        nonlocal current_tris, current_name
        if current_tris:
            groups.append((current_name, current_tris))
        current_tris = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if not parts:
                continue
            op = parts[0]
            if op == "v" and len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif op == "f":
                # Face indices; each token may be "i", "i/ti", "i/ti/ni", or "i//ni"
                idx = []
                for tok in parts[1:]:
                    v = tok.split("/")[0]
                    n = int(v)
                    if n < 0:                      # negative indices are relative
                        n = len(verts) + n + 1
                    idx.append(n - 1)              # OBJ is 1-based
                # Triangulate fan-style
                for i in range(1, len(idx) - 1):
                    current_tris.append((idx[0], idx[i], idx[i+1]))
            elif op in ("o", "g"):
                _flush()
                current_name = " ".join(parts[1:]) or current_name
        _flush()

    if not verts:
        return []

    vs = np.asarray(verts, dtype=np.float32)
    out = []
    for name, tris in groups:
        arr = np.asarray(tris, dtype=np.int32)
        if len(arr) == 0:
            continue
        out.append(vs[arr].astype(np.float32))
    return out


def write_obj(path, meshes, group_names=None):
    """Write a list of (N,3,3) triangle arrays to OBJ. Deduplicates verts
    per-mesh; each mesh becomes its own `o` group."""
    if not isinstance(meshes, list):
        meshes = [meshes]
    with open(path, "w") as f:
        f.write("# Generated by 3mf-stl-editor\n")
        global_base = 0
        for idx, m in enumerate(meshes):
            name = (group_names[idx] if group_names and idx < len(group_names)
                    else f"mesh_{idx+1}")
            flat = np.asarray(m, dtype=np.float32).reshape(-1, 3)
            unique, inverse = np.unique(flat, axis=0, return_inverse=True)
            tri_idx = inverse.reshape(-1, 3)

            f.write(f"o {name}\n")
            for v in unique:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for t in tri_idx:
                # OBJ is 1-based and counts globally across the file
                f.write(f"f {int(t[0])+1+global_base} {int(t[1])+1+global_base} {int(t[2])+1+global_base}\n")
            global_base += len(unique)


# ──────────────────────────────────────────────────────────────────────────────
#  PLY
# ──────────────────────────────────────────────────────────────────────────────

def read_ply(path):
    """Read a PLY file (ASCII or binary LE). Returns one-entry list for
    uniformity with other readers."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            return []
        header, fmt = [], "ascii"
        n_verts = n_faces = 0
        vert_props = []            # list of (name, dtype_str)
        face_list_type = None
        reading_faces = False
        while True:
            line = f.readline().strip()
            header.append(line)
            if line.startswith(b"format"):
                if b"binary_little_endian" in line: fmt = "binary_le"
                elif b"binary_big_endian" in line:  fmt = "binary_be"
                else:                                fmt = "ascii"
            elif line.startswith(b"element vertex"):
                n_verts = int(line.split()[-1])
                reading_faces = False
            elif line.startswith(b"element face"):
                n_faces = int(line.split()[-1])
                reading_faces = True
            elif line.startswith(b"property") and not reading_faces:
                parts = line.split()
                vert_props.append((parts[-1].decode(), parts[1].decode()))
            elif line.startswith(b"property list") and reading_faces:
                parts = line.split()
                face_list_type = (parts[2].decode(), parts[3].decode())  # (count_type, index_type)
            elif line == b"end_header":
                break

        # We only care about x,y,z in the vertex block
        xyz_idx = [i for i,(n,_) in enumerate(vert_props) if n in ("x","y","z")]
        if len(xyz_idx) != 3:
            return []

        if fmt == "ascii":
            verts = np.zeros((n_verts, 3), dtype=np.float32)
            for i in range(n_verts):
                parts = f.readline().split()
                verts[i] = [float(parts[xyz_idx[0]]),
                            float(parts[xyz_idx[1]]),
                            float(parts[xyz_idx[2]])]
            tris = []
            for _ in range(n_faces):
                parts = f.readline().split()
                n = int(parts[0])
                idx = [int(x) for x in parts[1:1+n]]
                for i in range(1, n-1):
                    tris.append([idx[0], idx[i], idx[i+1]])
            tris = np.array(tris, dtype=np.int32)
        else:
            # Binary: build a dtype that matches the vertex props order, then slice xyz
            type_map = {"float": "<f4", "float32": "<f4", "float64": "<f8", "double": "<f8",
                        "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
                        "uchar": "u1", "char": "i1", "ushort": "<u2", "short": "<i2"}
            if fmt == "binary_be":
                type_map = {k: v.replace("<", ">") for k, v in type_map.items()}
            dt = np.dtype([(n, type_map.get(t, "<f4")) for n,t in vert_props])
            raw = f.read(n_verts * dt.itemsize)
            varr = np.frombuffer(raw, dtype=dt, count=n_verts)
            verts = np.stack([varr["x"], varr["y"], varr["z"]], axis=1).astype(np.float32)

            # Faces: variable-length list, parse one by one (slower but faces are <<< verts)
            cnt_type = type_map.get(face_list_type[0], "u1")
            idx_type = type_map.get(face_list_type[1], "<i4")
            cnt_size = np.dtype(cnt_type).itemsize
            idx_size = np.dtype(idx_type).itemsize
            tris = []
            for _ in range(n_faces):
                n = int(np.frombuffer(f.read(cnt_size), dtype=cnt_type)[0])
                idx = np.frombuffer(f.read(n * idx_size), dtype=idx_type).tolist()
                for i in range(1, n-1):
                    tris.append([idx[0], idx[i], idx[i+1]])
            tris = np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)

        if len(tris) == 0:
            return []
        return [verts[tris].astype(np.float32)]


def write_ply(path, meshes):
    """Write triangles to a binary little-endian PLY. Merges all meshes."""
    if not isinstance(meshes, list):
        meshes = [meshes]
    if not meshes:
        raise ValueError("No meshes to write")
    merged = np.concatenate(meshes, axis=0)
    flat = merged.reshape(-1, 3).astype(np.float32)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    tri_idx = inverse.reshape(-1, 3).astype(np.int32)

    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"comment Generated by 3mf-stl-editor\n")
        f.write(f"element vertex {len(unique)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(tri_idx)}\n".encode())
        f.write(b"property list uchar int vertex_indices\n")
        f.write(b"end_header\n")
        f.write(unique.astype("<f4").tobytes())
        # Face: 1 byte count (=3) + 3 int32 indices per face
        rec = np.empty(len(tri_idx), dtype=[("n", "u1"), ("v", "<i4", 3)])
        rec["n"] = 3
        rec["v"] = tri_idx
        f.write(rec.tobytes())


# ──────────────────────────────────────────────────────────────────────────────
#  GLB (GLTF binary) — reader only
# ──────────────────────────────────────────────────────────────────────────────

def read_glb(path):
    """Read a .glb file. Returns list of world-space triangle arrays, one per
    mesh primitive. Handles the common GLTF2 accessor types (FLOAT VEC3 for
    POSITION, UNSIGNED_SHORT / UNSIGNED_INT for indices). Applies node
    transforms from the first scene."""
    with open(path, "rb") as f:
        magic, version, length = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            raise ValueError("Not a GLB file")
        # Chunk 0: JSON
        clen, ctype = struct.unpack("<I4s", f.read(8))
        json_bytes = f.read(clen)
        gltf = json.loads(json_bytes)
        # Chunk 1: BIN
        bin_data = b""
        if f.tell() < length:
            clen, ctype = struct.unpack("<I4s", f.read(8))
            bin_data = f.read(clen)

    buffer_views = gltf.get("bufferViews", [])
    accessors    = gltf.get("accessors", [])
    meshes       = gltf.get("meshes", [])
    nodes        = gltf.get("nodes", [])
    scenes       = gltf.get("scenes", [])

    _GL_TYPE = {5120:"i1", 5121:"u1", 5122:"<i2", 5123:"<u2", 5125:"<u4", 5126:"<f4"}
    _N_COMP  = {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4, "MAT4":16}

    def read_acc(idx):
        a = accessors[idx]
        bv = buffer_views[a["bufferView"]]
        offset = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = _GL_TYPE[a["componentType"]]
        n = a["count"] * _N_COMP[a["type"]]
        arr = np.frombuffer(bin_data, dtype=dt, count=n, offset=offset)
        if a["type"] != "SCALAR":
            arr = arr.reshape(-1, _N_COMP[a["type"]])
        return np.asarray(arr).copy()

    def compose_trs(node):
        m = np.eye(4, dtype=np.float64)
        if "matrix" in node:
            m = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
        else:
            t = node.get("translation", [0,0,0])
            r = node.get("rotation", [0,0,0,1])
            s = node.get("scale", [1,1,1])
            S = np.diag([s[0], s[1], s[2], 1.0])
            # Quaternion to matrix
            x,y,z,w = r
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),   0],
                [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),   0],
                [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y), 0],
                [0,0,0,1],
            ])
            T = np.eye(4); T[:3, 3] = t
            m = T @ R @ S
        return m

    def walk(node_idx, parent_mat, out):
        node = nodes[node_idx]
        m = parent_mat @ compose_trs(node)
        if "mesh" in node:
            for prim in meshes[node["mesh"]]["primitives"]:
                pos_acc = prim["attributes"].get("POSITION")
                if pos_acc is None:
                    continue
                verts = read_acc(pos_acc).astype(np.float32)
                if "indices" in prim:
                    idx = read_acc(prim["indices"]).reshape(-1).astype(np.int64)
                    tri_idx = idx.reshape(-1, 3)
                else:
                    tri_idx = np.arange(len(verts)).reshape(-1, 3)
                # Apply node transform
                ones = np.ones((len(verts), 1))
                world = (m @ np.hstack([verts.astype(np.float64), ones]).T).T[:, :3]
                out.append(world[tri_idx].astype(np.float32))
        for c in node.get("children", []):
            walk(c, m, out)

    result = []
    scene = gltf.get("scene", 0)
    if scenes:
        for ni in scenes[scene].get("nodes", []):
            walk(ni, np.eye(4), result)
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Uniform entry point
# ──────────────────────────────────────────────────────────────────────────────

def read(path):
    """Load any supported mesh format; returns list of (N,3,3) float32 arrays.
    Raises ValueError for CAD native formats (STEP/IGES/etc)."""
    info = sniff_format(path)
    if info["kind"] == "cad":
        raise ValueError(
            f"{info['detail']}. This is a CAD native format — not a mesh. "
            f"Export to STL / OBJ / 3MF from your CAD program first."
        )
    if info["kind"] == "unknown":
        raise ValueError(f"Unknown file type: {info['ext']!r}")

    ext = info["ext"]
    if ext == ".stl" or ext == ".3mf":
        # Delegate to the established readers
        from _3mf_reader import read_stl, read_3mf
        return read_stl(path) if ext == ".stl" else read_3mf(path)
    if ext == ".obj":
        return read_obj(path)
    if ext == ".ply":
        return read_ply(path)
    if ext == ".glb":
        return read_glb(path)
    raise ValueError(f"Reader not implemented: {ext}")


def write_glb(path, meshes):
    """Write meshes to a GLB (GLTF binary) file.
    Produces a valid GLTF 2.0 binary with a single mesh per input array,
    indexed geometry, and no materials/textures. Compatible with Sketchfab,
    Blender, Three.js, Babylon.js, and most web/AR viewers."""
    import struct, json as _json

    if not isinstance(meshes, list):
        meshes = [meshes]

    # Build binary buffer (all vertex + index data concatenated)
    buffer_views = []
    accessors = []
    primitives = []
    bin_chunks = []
    byte_offset = 0

    for m in meshes:
        flat = np.asarray(m, dtype=np.float32).reshape(-1, 3)
        unique, inverse = np.unique(flat, axis=0, return_inverse=True)
        tri_idx = inverse.reshape(-1, 3).astype(np.uint32)

        # Vertex positions
        verts_bytes = unique.astype(np.float32).tobytes()
        verts_len = len(verts_bytes)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset,
                              "byteLength": verts_len, "target": 34962})  # ARRAY_BUFFER
        vert_bv = len(buffer_views) - 1
        mins = unique.min(0).tolist()
        maxs = unique.max(0).tolist()
        accessors.append({"bufferView": vert_bv, "componentType": 5126,  # FLOAT
                          "count": len(unique), "type": "VEC3",
                          "min": mins, "max": maxs})
        vert_acc = len(accessors) - 1
        bin_chunks.append(verts_bytes)
        byte_offset += verts_len

        # Indices
        idx_bytes = tri_idx.flatten().astype(np.uint32).tobytes()
        idx_len = len(idx_bytes)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset,
                              "byteLength": idx_len, "target": 34963})  # ELEMENT_ARRAY_BUFFER
        idx_bv = len(buffer_views) - 1
        accessors.append({"bufferView": idx_bv, "componentType": 5125,  # UNSIGNED_INT
                          "count": int(len(tri_idx) * 3), "type": "SCALAR"})
        idx_acc = len(accessors) - 1
        bin_chunks.append(idx_bytes)
        byte_offset += idx_len

        primitives.append({"attributes": {"POSITION": vert_acc}, "indices": idx_acc})

    bin_data = b"".join(bin_chunks)
    # Pad to 4-byte boundary
    if len(bin_data) % 4:
        bin_data += b"\x00" * (4 - len(bin_data) % 4)

    gltf = {
        "asset": {"version": "2.0", "generator": "3d-file-toolkit"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(primitives)))}],
        "nodes": [{"mesh": i} for i in range(len(primitives))],
        "meshes": [{"primitives": [p]} for p in primitives],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_data)}],
    }

    json_bytes = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Pad JSON to 4-byte boundary with spaces
    if len(json_bytes) % 4:
        json_bytes += b" " * (4 - len(json_bytes) % 4)

    # GLB structure: 12-byte header + JSON chunk + BIN chunk
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    with open(path, "wb") as f:
        f.write(b"glTF")                                    # magic
        f.write(struct.pack("<II", 2, total_len))           # version, total length
        f.write(struct.pack("<I", len(json_bytes)))         # JSON chunk length
        f.write(b"JSON")                                    # JSON chunk type
        f.write(json_bytes)
        f.write(struct.pack("<I", len(bin_data)))           # BIN chunk length
        f.write(b"BIN\x00")                                # BIN chunk type
        f.write(bin_data)


def write(path, meshes):
    """Write meshes to any supported output format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        from _3mf_reader import write_stl_binary
        merged = np.concatenate(meshes, axis=0) if len(meshes) > 1 else meshes[0]
        write_stl_binary(path, merged, header_text="3d-file-toolkit")
        return
    if ext == ".obj":
        write_obj(path, meshes)
        return
    if ext == ".ply":
        write_ply(path, meshes)
        return
    if ext == ".glb":
        write_glb(path, meshes)
        return
    if ext == ".3mf":
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from convert import build_3mf_model_xml, write_3mf
        model_xml = build_3mf_model_xml(meshes if isinstance(meshes, list) else [meshes])
        write_3mf(path, model_xml)
        return
    raise ValueError(f"Writer not implemented: {ext}")


__all__ = ["sniff_format", "read", "write",
           "read_obj", "write_obj", "read_ply", "write_ply", "read_glb", "write_glb",
           "MESH_EXTS", "CAD_EXTS"]
