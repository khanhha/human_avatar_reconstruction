import bmesh
import mathutils
import bpy
import numpy as np

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

#note: this script just runs in object mode. not sure why
#note: this script doesn't work because the face order is different between two objects. not sure why. Blender reorder face order when it export to obj file

ob = bpy.data.objects['VictoriaMesh']
ob_uvmap = ob.data.uv_layers.active.data 
target_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_template_textured_warped.obj'
tar_mesh = import_mesh_tex_obj(target_path)
verts_uv = tar_mesh['vt']
faces_uv = tar_mesh['ft']
# Loops per face
for f_idx, face in enumerate(ob.data.polygons):
    cnt = 0
    for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
        tar_face = faces_uv[f_idx]
        print(tar_face)
        break                     
        ob_uvmap[loop_idx].uv[0] = verts_uv[tar_face[cnt], 0]
        ob_uvmap[loop_idx].uv[1] = verts_uv[tar_face[cnt], 1]        
        cnt += 1
        #print("face idx: %i, vert idx: %i, uvs: %f, %f" % (face.index, vert_idx, uv_coords.x, uv_coords.y))
