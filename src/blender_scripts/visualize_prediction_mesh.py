import bpy
import numpy as np
import bmesh

def import_mesh_tex_obj(fpath):
    verts = []
    verts_tex = []
    faces =  []
    faces_tex =  []
    with open(fpath, 'r') as obj:
        file = obj.read()
        lines = file.splitlines()
        for line in lines:
            elem = line.split()
            if elem:
                if elem[0] == 'v':
                    verts.append((float(elem[1]), float(elem[2]), float(elem[3])))
                elif elem[0] == 'vt':
                    verts_tex.append((float(elem[1]), float(elem[2])))
                elif elem[0] == 'vn' or elem[0] == 'vp':
                    raise Exception('unsupported format')
                elif elem[0] == 'f':
                    f = []
                    ft = []
                    for v_idx_str in elem[1:]:
                        v_idx = v_idx_str.split('/')
                        if len(v_idx) != 2:
                            raise Exception('unsupported format')
                        f.append(int(v_idx[0])-1)
                        ft.append(int(v_idx[1])-1)
                    faces.append(f)
                    faces_tex.append(ft)
    mesh = {}
    mesh['v']   = np.array(verts)
    mesh['vt']  = np.array(verts_tex)
    mesh['f']   = faces
    mesh['ft']  = faces_tex
    return mesh

def find_grp_verts(obj, grp_name_hint):
    grp_verts = []
    for idx, v in enumerate(obj.data.vertices):
        for vgrp in v.groups:
            grp_name = obj.vertex_groups[vgrp.group].name
            if grp_name_hint == grp_name:
                grp_verts.append(idx)        
    return np.array(grp_verts)
            
def collect_measurement_vert_groups(obj):    
    grp_names = set()
    for grp in obj.vertex_groups:
        if 'verts_circ_' in grp.name:
            grp_names.add(grp.name)
    
    vert_groups = dict()
    for grp_name in grp_names:
        verts = find_grp_verts(obj, grp_name)
        assert len(verts) > 0, grp_name
        vert_groups[grp_name] = verts
    
    return vert_groups

def set_mesh(obj, verts):
    mesh = obj.data
    for i, v in enumerate(mesh.vertices):
        v.co[:] = verts[i,:]

def highlight_vert_groups(obj, vgroups):
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    for v in bm.verts:
        v.select = False
        
    for name, vidxs in vgroups.items():
        for idx in vidxs:
            bm.verts[idx].select = True
    
    
#dir  = "/media/F/projects/Oh/data/body_test_data/body_designer/result_posevar/"
#mesh_path = dir + 'designer_0_front_designer_0_seg_hd_side.obj'
#mesh_path = dir + 'cory_1933_front_cory_1933_seg_hd_1_side.obj'
dir = "/media/D1/data_1/projects/Oh/data/3d_human/test_data/body_designer/result_posevar/"
mesh_path = dir + "cory_1933_front_designer_0_face.obj"
#mesh_path = dir + "designer_0_front_designer_0_face.obj"
    
objmesh = import_mesh_tex_obj(mesh_path)
verts = objmesh['v']

obj = bpy.data.objects['VictoriaMeshMeasureViz']
set_mesh(obj, verts)

vgroups = collect_measurement_vert_groups(obj)

highlight_vert_groups(obj, vgroups)
    





