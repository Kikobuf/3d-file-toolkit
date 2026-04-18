"""Generate minimal test fixtures: a 10mm cube in both STL and 3MF formats.

Run once to create:
  - cube.stl               (simple binary STL, 12 tris)
  - cube_flat.3mf          (3MF with inline <mesh> in main model)
  - cube_external.3mf      (3MF with <component> pointing to 3D/Objects/*.model,
                            the Bambu/Orca pattern that broke v1 of transform.py)

These are the canonical regression fixtures for the test suite.
"""
import os, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from _3mf_reader import write_stl_binary


def cube_tris(size=10.0, origin=(0, 0, 0)):
    """Return (12, 3, 3) triangle array for an axis-aligned cube."""
    s = size
    ox, oy, oz = origin
    v = np.array([
        [ox,   oy,   oz],   # 0
        [ox+s, oy,   oz],   # 1
        [ox+s, oy+s, oz],   # 2
        [ox,   oy+s, oz],   # 3
        [ox,   oy,   oz+s], # 4
        [ox+s, oy,   oz+s], # 5
        [ox+s, oy+s, oz+s], # 6
        [ox,   oy+s, oz+s], # 7
    ], dtype=np.float32)
    # 6 faces, 2 tris each, CCW from outside
    faces = [
        (0,1,2),(0,2,3),     # bottom -z
        (4,6,5),(4,7,6),     # top    +z
        (0,4,5),(0,5,1),     # front  -y
        (2,6,7),(2,7,3),     # back   +y
        (0,3,7),(0,7,4),     # left   -x
        (1,5,6),(1,6,2),     # right  +x
    ]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float32)


# ---- 3MF builders ---------------------------------------------------------

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model" Id="rel0"/>
</Relationships>"""


def _mesh_xml_indexed(verts, tri_idx, indent="      "):
    """Serialize vertex+triangle XML for a cube mesh (8 unique verts, 12 tris)."""
    vx = "\n".join(f'{indent}  <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>'
                   for v in verts)
    tx = "\n".join(f'{indent}  <triangle v1="{t[0]}" v2="{t[1]}" v3="{t[2]}"/>'
                   for t in tri_idx)
    return (f'{indent}<mesh>\n'
            f'{indent} <vertices>\n{vx}\n{indent} </vertices>\n'
            f'{indent} <triangles>\n{tx}\n{indent} </triangles>\n'
            f'{indent}</mesh>')


def build_flat_3mf(path):
    """Inline-mesh 3MF — mesh lives directly in the main model."""
    verts = np.array([
        [0,0,0],[10,0,0],[10,10,0],[0,10,0],
        [0,0,10],[10,0,10],[10,10,10],[0,10,10],
    ], dtype=np.float32)
    tri_idx = np.array([
        [0,1,2],[0,2,3],
        [4,6,5],[4,7,6],
        [0,4,5],[0,5,1],
        [2,6,7],[2,7,3],
        [0,3,7],[0,7,4],
        [1,5,6],[1,6,2],
    ])
    mesh_xml = _mesh_xml_indexed(verts, tri_idx)
    model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" unit="millimeter">
 <metadata name="Title">Test Cube (flat)</metadata>
 <resources>
  <object id="1" type="model" name="cube">
{mesh_xml}
  </object>
 </resources>
 <build>
  <item objectid="1"/>
 </build>
</model>
'''
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("3D/3dmodel.model", model)


def build_external_3mf(path):
    """Bambu-style 3MF: main model has <component> pointing to 3D/Objects/object_1.model.
    Build item translates the object to (5, 5, 0). Critical regression fixture."""
    verts = np.array([
        [0,0,0],[10,0,0],[10,10,0],[0,10,0],
        [0,0,10],[10,0,10],[10,10,10],[0,10,10],
    ], dtype=np.float32)
    tri_idx = np.array([
        [0,1,2],[0,2,3],
        [4,6,5],[4,7,6],
        [0,4,5],[0,5,1],
        [2,6,7],[2,7,3],
        [0,3,7],[0,7,4],
        [1,5,6],[1,6,2],
    ])
    mesh_xml = _mesh_xml_indexed(verts, tri_idx, indent="   ")

    sub_model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
       unit="millimeter" requiredextensions="p">
 <resources>
  <object id="1" p:UUID="00000001-cube-cube-cube-000000000001" type="model" name="cube">
{mesh_xml}
  </object>
 </resources>
</model>
'''

    # Main model: object 2 is a component wrapper, referenced by build item
    # Build item translates to (5, 5, 0). No rotation, identity component transform.
    main_model = '''<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
       unit="millimeter" requiredextensions="p">
 <metadata name="Title">Test Cube (external components)</metadata>
 <metadata name="Application">fixture-builder</metadata>
 <resources>
  <object id="2" p:UUID="00000002-wrap-wrap-wrap-000000000002" type="model">
   <components>
    <component objectid="1" p:path="/3D/Objects/object_1.model"
               p:UUID="00010000-comp-comp-comp-000000000001"
               transform="1 0 0 0 1 0 0 0 1 0 0 0"/>
   </components>
  </object>
 </resources>
 <build p:UUID="00000003-bild-bild-bild-000000000003">
  <item objectid="2" p:UUID="00000004-item-item-item-000000000004"
        transform="1 0 0 0 1 0 0 0 1 5 5 0" printable="1"/>
 </build>
</model>
'''

    sub_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
'''

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("3D/3dmodel.model", main_model)
        zf.writestr("3D/_rels/3dmodel.model.rels", sub_rels)
        zf.writestr("3D/Objects/object_1.model", sub_model)


def main():
    fdir = os.path.dirname(os.path.abspath(__file__))
    stl_path          = os.path.join(fdir, "cube.stl")
    flat_3mf_path     = os.path.join(fdir, "cube_flat.3mf")
    external_3mf_path = os.path.join(fdir, "cube_external.3mf")

    write_stl_binary(stl_path, cube_tris(size=10.0, origin=(0,0,0)),
                     header_text="Test cube fixture")
    build_flat_3mf(flat_3mf_path)
    build_external_3mf(external_3mf_path)

    for p in (stl_path, flat_3mf_path, external_3mf_path):
        print(f"  {os.path.getsize(p):>8,} B  {os.path.relpath(p, fdir)}")


if __name__ == "__main__":
    main()
