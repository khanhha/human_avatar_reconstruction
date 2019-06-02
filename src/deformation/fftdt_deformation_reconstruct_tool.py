import argparse
import pickle
from deformation import ffdt_deformation_lib as df
from common.obj_util import import_mesh_obj, export_mesh
from copy import deepcopy
from deformation.ffdt_deformation_lib import TemplateMeshDeform
import numpy as np
from pathlib import Path
from os.path import join

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

    #try:
    tpl_verts, tpl_faces = import_mesh_obj(fpath=tpl_path)

    ctl_df_verts, ctl_df_faces = import_mesh_obj(fpath=ctl_path)

    mean = np.mean(ctl_df_verts, axis=0)
    ctl_df_verts = ctl_df_verts - mean
    ctl_df_verts *= 0.02

    for idx, tris in enumerate(ctl_df_faces):
        assert (len(tris) == 3), f'face {idx} with len of {len(tris)} is not a triangle'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        vert_UVWs = data['template_vert_UVW']
        vert_weights = data['template_vert_weight']
        vert_effect_idxs = data['template_vert_effect_idxs']

        deform = TemplateMeshDeform(effective_range=4, use_mean_rad=False)
        deform.set_meshes(ctl_verts=ctl_df_verts, ctl_tris=ctl_df_faces, tpl_verts=tpl_verts, tpl_faces=tpl_faces)
        deform.set_parameterization(vert_tri_UVWs=vert_UVWs, vert_tri_weights=vert_weights,
                                    vert_effect_tri_idxs=vert_effect_idxs)

        del vert_UVWs
        del vert_weights
        del vert_effect_idxs
        del data

    tpl_df_verts = deform.deform(ctl_df_verts)

    print(f'output deformed template mesh to file {out_path}')
    export_mesh(out_path, tpl_df_verts, tpl_faces)

    out_path = join(*[Path(out_path).parent, f'{Path(out_path).stem}_ground_truth.obj'])
    export_mesh(out_path, ctl_df_verts, ctl_df_faces)


    # except Exception as exp:
    #     print('opp, something wrong: ', exp)
    #     exit()






