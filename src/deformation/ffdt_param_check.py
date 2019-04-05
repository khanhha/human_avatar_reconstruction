import argparse
import pickle
import numpy as np
from common.obj_util import import_mesh, export_mesh
from copy import deepcopy
from pathlib import Path

import os
def merge_two_mesh(verts_0, tris_0, verts_1, tris_1):
    nverts_0 = verts_0.shape[0]
    verts = np.concatenate([verts_0, verts_1], axis=0)

    tris_1_new = deepcopy((tris_1))
    for i in range(len(tris_1_new)):
        s = len(tris_1_new[i])
        for j in range(s):
            tris_1_new[i][j] += nverts_0

    return verts, tris_0+tris_1_new

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--param_file", required=True, help="parameterization file")
    ap.add_argument("-ctl", "--control_mesh", required=True, help="control mesh")
    ap.add_argument("-tpl", "--template_mesh", required=True, help="template mesh")
    ap.add_argument("-ndir", "--neighbor_dir", required=True, help="output dir to export neighbor information")
    ap.add_argument("-pnt_mesh", "--pointer_mesh", required=True, help="pointer mesh")

    args = vars(ap.parse_args())
    param_path = args['param_file']
    ctl_mesh_path = args['control_mesh']
    tpl_mesh_path = args['template_mesh']
    pnt_mesh_path = args['pointer_mesh']
    ndir = args['neighbor_dir']

    os.makedirs(ndir, exist_ok=True)
    for path in Path(ndir).glob('*.obj'):
        os.remove(path)

    # for path in Path('../').glob('*.*'):
    #     print(path)

    with open(param_path, 'rb') as file:
        param_data = pickle.load(file=file)
        vert_effect_idxs = param_data['template_vert_effect_idxs']

    ctl_mesh_verts, ctl_mesh_tris = import_mesh(ctl_mesh_path)
    tpl_mesh_verts, _ = import_mesh(tpl_mesh_path)
    pnt_mesh_verts, pnt_mesh_tris = import_mesh(pnt_mesh_path)


    # v_idxs = [2945, 4373, 14, 69, 7322, 7993, 6941, 7946, 7750, 6445, 8321, 7634, 6145, 1042, 140, 2965, 4419, 3980, 5606, 6017]
    # v_idxs = v_idxs + [13825, 12348, 6924, 7929, 12896, 14373]
    # v_idxs = v_idxs + [36262, 36241, 35458, 36115, 35334, 38022, 38719]
    v_idxs = [23383]
    for v_idx in v_idxs:
        #pick effect triangles
        neighbor_tris = [ctl_mesh_tris[t_idx] for t_idx in vert_effect_idxs[v_idx ]]

        #add a pointer for marking the location of the vertex v_idx
        #the pointer mesh center should be at (0,0,0)
        v_co = tpl_mesh_verts[v_idx, :]
        pnt_verts = deepcopy(pnt_mesh_verts)
        pnt_verts = pnt_verts + v_co

        neighbor_verts, neighbor_tris = merge_two_mesh(ctl_mesh_verts, neighbor_tris, pnt_verts, pnt_mesh_tris)
        print(f'exported neighbor mesh: {ndir}/neighbour_tri_of_{v_idx }.obj')
        export_mesh(f'{ndir}/neighbour_tri_of_{v_idx }.obj', verts=neighbor_verts, faces=neighbor_tris)

