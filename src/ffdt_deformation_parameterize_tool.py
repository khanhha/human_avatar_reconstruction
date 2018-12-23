import argparse
import pickle
import ffdt_deformation_lib as df
import numpy as np
import scipy.stats as stats
from obj_util import import_mesh

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ctl", "--control_mesh_path", required=True, help="control mesh path")
    ap.add_argument("-tpl", "--template_mesh_path", required=True, help="template mesh path")
    ap.add_argument("-o", "--out_path", required=True, help="data output path")
    ap.add_argument("-g", "--global", required=True, help="global weight")

    args = vars(ap.parse_args())

    ctl_out_path = args['control_mesh_path']
    tpl_out_path = args['template_mesh_path']
    out_path = args['out_path']
    is_global = int(args['global']) > 0


    ctl_verts, ctl_faces = import_mesh(ctl_out_path)
    for idx, tri in enumerate(ctl_faces):
        assert len(tri) == 3, f'face {idx} is not a triangle, n_vert = {len(tri)}'

    tpl_verts, tpl_faces = import_mesh(tpl_out_path)

    print('control  mesh:            nverts = {0}, nfaces = {1}'.format(ctl_verts.shape[0], len(ctl_faces)))
    print('template mesh (victoria): nverts = {0}, nfaces = {1}'.format(tpl_verts.shape[0], len(tpl_faces)))

    print(f'\nstart calculating local basis (U,V,W) for each control mesh triangle \m')
    ctl_tri_bs = df.calc_triangle_local_basis(ctl_verts, ctl_faces)
    print(f'\tfinish local basis calculation')

    if is_global:
        print(f'\nstart calculating weights')
        vert_effect_idxs, vert_weights = df.calc_vertex_weigth_control_mesh_global(tpl_verts, ctl_verts, ctl_faces, effective_range_factor=3)
        lens = np.array([len(idxs) for idxs in vert_effect_idxs])
        stat = stats.describe(lens)
        print(f'\tfinish weight calculation')
        print(f'\tneighbor size statistics: mean number of neighbor, variance number of neighbor')
        print(f'\t{stat}')
    else:
        print('not support local parameterization currently')
        exit()
        #ctl_f_body_parts = data['control_mesh_face_body_parts']
        #tpl_v_body_parts = data['template_vert_body_parts']
        #body_part_dict = data['body_part_dict']
        #vert_effect_idxs, vert_weights = df.calc_vertex_weigth_control_mesh_local(tpl_mesh['verts'], ctl_mesh['verts'], ctl_mesh['faces'], tpl_v_body_parts, ctl_f_body_parts)

    print(f'\nstart calculating relative coordinates of template mesh vertices with respect to basis of neighbour control mesh triangle')
    vert_UVW = df.parameterize(tpl_verts, vert_effect_idxs, ctl_tri_bs)
    print(f'\tfinish relative coordinate calculation')

    w_data = {}
    w_data['template_vert_UVW'] = vert_UVW
    w_data['template_vert_weight'] = vert_weights
    w_data['template_vert_effect_idxs'] = vert_effect_idxs
    w_data['control_mesh_tri_basis'] = ctl_tri_bs

    with open(out_path, 'wb') as f:
        pickle.dump(w_data, f)
    print(f'\noutput parameterization to file {out_path}')


