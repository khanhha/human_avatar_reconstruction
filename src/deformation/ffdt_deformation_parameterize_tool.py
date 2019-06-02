import argparse
import pickle
from deformation import ffdt_deformation_lib as df
import numpy as np
import scipy.stats as stats
from common.obj_util import import_mesh_obj

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ctl", "--control_mesh_path", required=True, help="control mesh path")
    ap.add_argument("-tpl", "--template_mesh_path", required=True, help="template mesh path")
    ap.add_argument("-o", "--out_dir", required=True, help="data output dir")
    ap.add_argument("-mode", required=True, help="mode")
    ap.add_argument("-cdd_tris", required=False, type=str, default='')
    ap.add_argument("-e", "--effective_range", required=False, default=3)
    ap.add_argument("-use_mean_rad", "--use_mean_radius", required=False, default=0)
    ap.add_argument("-out_name_id", required=False, type=str, default='parameterization')

    args = vars(ap.parse_args())

    ctl_out_path = args['control_mesh_path']
    tpl_out_path = args['template_mesh_path']
    OUT_DIR = args['out_dir']
    mode = args['mode']
    effecttive_range = float(args['effective_range'])
    use_mean_rad = int(args['use_mean_radius']) > 0

    ctl_verts, ctl_faces = import_mesh_obj(ctl_out_path)
    for idx, tri in enumerate(ctl_faces):
        assert len(tri) == 3, f'face {idx} is not a triangle, n_vert = {len(tri)}'

    tpl_verts, tpl_faces = import_mesh_obj(tpl_out_path)

    print('control  mesh:            nverts = {0}, nfaces = {1}'.format(ctl_verts.shape[0], len(ctl_faces)))
    print('template mesh (victoria): nverts = {0}, nfaces = {1}'.format(tpl_verts.shape[0], len(tpl_faces)))

    print(f'\nstart calculating local basis (U,V,W) for each control mesh triangle \m')
    ctl_tri_bs = df.calc_triangle_local_basis(ctl_verts, ctl_faces)
    print(f'\tfinish local basis calculation')

    if mode == 'global':
        print(f'\nstart calculating weights, mode = {mode}')
        print(f'\n\teffective range = {effecttive_range}, use_mean_radius = {use_mean_rad}')
        vert_effect_idxs, vert_weights = df.calc_vertex_weigth_control_mesh_global(tpl_verts, ctl_verts, ctl_faces, effective_range_factor=effecttive_range, use_mean_tri_radius=use_mean_rad)
        lens = np.array([len(idxs) for idxs in vert_effect_idxs])
        stat = stats.describe(lens)
        print(f'\tfinish weight calculation')
        print(f'\tneighbor size statistics: mean number of neighbor, variance number of neighbor')
        print(f'\t{stat}')
    elif mode == 'fixed_range':
        print(f'\nstart calculating weights, mode = {mode}')
        cdd_tris_path = args['cdd_tris']
        with open(cdd_tris_path, 'rb') as file:
            vert_effect_cdd_idxs = pickle.load(file=file)
        vert_effect_idxs, vert_weights = df.calc_vert_weight(tpl_verts, vert_effect_cdd_idxs, ctl_verts, ctl_faces, effecttive_range)
        print(f'\tfinish weight calculation')
    else:
        print('not support local parameterization on currently')
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

    print(f'\nsaving data')
    id = args['out_name_id']
    with open(f'{OUT_DIR}/{mode}_{id}.pkl', 'wb') as f:
        pickle.dump(w_data, f, protocol=4)
    print(f'\n\toutput parameterization to file {OUT_DIR}')


