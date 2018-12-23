import argparse
import pickle
import ffdt_deformation_lib as df
from obj_util import import_mesh, export_mesh
from copy import deepcopy

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template_mesh", required=True, help="template mesh to be deformed")
    ap.add_argument("-d", "--deformed_ctr_mesh", required=True, help="deformed control mesh")
    ap.add_argument("-p", "--parameterization", required=True, help="parameterization data of the template mesh with respect to the org control mesh")
    ap.add_argument("-o", "--output_deformed_template_mesh", required=True, help="output deformed template mesh")

    args = vars(ap.parse_args())

    tpl_path  = args['template_mesh']
    ctl_path  = args['deformed_ctr_mesh']
    data_path = args['parameterization']
    out_path  = args['output_deformed_template_mesh']

    try:
        tpl_verts, tpl_faces = import_mesh(fpath=tpl_path)

        ctl_df_verts, ctl_df_faces = import_mesh(fpath=ctl_path)
        for idx, tris in enumerate(ctl_df_faces):
            assert (len(tris) == 3), f'face {idx} with len of {len(tris)} is not a triangle'

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            ctl_tri_bs = data['control_mesh_tri_basis']
            vert_UVWs = data['template_vert_UVW']
            vert_weights = data['template_vert_weight']
            vert_effect_idxs = data['template_vert_effect_idxs']

        print(f'start calculating basis for each triangle of deformed control mesh')
        ctl_df_basis = df.calc_triangle_local_basis(ctl_df_verts, ctl_df_faces)
        print(f'\tfinish basis caculattion')

        print(f'start reconstructing a new template mesh from new basis and parameterization')
        tpl_df_verts = deepcopy(tpl_verts)
        df.deform_template_mesh(tpl_df_verts, vert_effect_idxs, vert_weights, vert_UVWs, ctl_df_basis)
        print(f'\tfinish reconstruction')

        print(f'output deformed template mesh to file {out_path}')
        export_mesh(out_path, tpl_df_verts, tpl_faces)

    except Exception as exp:
        print('opp, something wrong: ', exp)
        exit()






