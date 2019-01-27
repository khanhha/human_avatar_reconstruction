from control_mesh import  ControlMeshPredictor
from deformation.ffdt_deformation_lib import TemplateMeshDeform
from common.obj_util import export_mesh
import numpy as np
from pathlib import Path
import argparse
import pickle
import shutil
import os
from tqdm import *
import multiprocessing
from functools import partial
import gc

def util_reconstruct_single_mesh(record, OUT_DIR, predictor, deformer):
    idx = record[0]
    mdata_path = record[1]

    if 'CSR0309A' not in mdata_path.name:
         return

    if idx % 20 == 0:
        print(f'{idx} - {mdata_path.name}')

    # load 2d measurements
    mdata = np.load(mdata_path).item()

    seg_dst_f = mdata['landmark_segment_dst_f']
    seg_dst_s = mdata['landmark_segment_dst_s']
    seg_locs_s = mdata['landmark_segment_location_s']
    seg_locs_f = mdata['landmark_segment_location_f']
    measurements = mdata['measurement']
    height = measurements['Height']

    ctl_tri_mesh = predictor.predict(seg_dst_f, seg_dst_s, seg_locs_s, seg_locs_f, height)

    # out_path = f'{OUT_DIR}{mdata_path.stem}_ctl_tri.obj'
    # export_mesh(out_path, verts=ctl_tri_mesh['verts'], faces=ctl_tri_mesh['faces'])
    out_path = f'{OUT_DIR}{mdata_path.stem}_ctl_quad.obj'
    export_mesh(out_path, verts=ctl_tri_mesh['verts'], faces=ctl_mesh_quad_dom['faces'])

    # tpl_new_verts, tpl_faces = deform.deform(ctl_tri_mesh['verts'])
    # out_path = f'{OUT_DIR}{mdata_path.stem}_tpl_deformed.obj'
    # export_mesh(out_path, verts=tpl_new_verts, faces=tpl_faces)
    gc.collect()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    ap.add_argument("-m", "--measure_dir", required=True, help="measurement 2d data directory")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    ap.add_argument("-w", "--weight", required=True, help="deform based on weight")
    ap.add_argument("-mo", "--model_dir", required=True, help="deform based on weight")
    ap.add_argument("-np", type=int, default = 1, help='')

    args = vars(ap.parse_args())

    in_path = args['input']
    weight_path = args['weight']
    M_DIR = args['measure_dir']
    MODEL_DIR = args['model_dir']
    OUT_DIR = args['out_dir'] + '/'
    n_process = args.get('np')

    # shutil.rmtree(OUT_DIR, ignore_errors=True)
    # os.makedirs(OUT_DIR)

    with open(in_path, 'rb') as f:
        data = pickle.load(f)

        ctl_mesh = data['control_mesh']
        ctl_mesh_quad_dom = data['control_mesh_quad_dom']
        slc_id_vert_idxs = data['slice_vert_idxs']
        slc_id_locs = data['slice_locs']
        ctl_sym_vert_pairs = data['control_mesh_symmetric_vert_pairs']

        tpl_mesh = data['template_mesh']
        tpl_height = data['template_height']
        tpl_sym_vert_pairs = data['template_symmetric_vert_pairs']

        arm_3d_tpl = data['arm_bone_locs']

        #load slice predictor
        predictor = ControlMeshPredictor(MODEL_DIR=MODEL_DIR)
        predictor.set_control_mesh(ctl_mesh=ctl_mesh, slc_id_vert_idxs=slc_id_vert_idxs, slc_id_locs=slc_id_locs, ctl_sym_vert_pairs=ctl_sym_vert_pairs, arm_3d_tpl=arm_3d_tpl)
        predictor.set_template_mesh(tpl_mesh=tpl_mesh, tpl_height=tpl_height)

    #load deform
    with open(weight_path, 'rb') as f:
        data = pickle.load(f)
        ctl_tri_bs = data['control_mesh_tri_basis']
        vert_UVWs = data['template_vert_UVW']
        vert_weights = data['template_vert_weight']
        vert_effect_idxs = data['template_vert_effect_idxs']

        deform = TemplateMeshDeform(effective_range=4, use_mean_rad=False)
        deform.set_meshes(ctl_verts=ctl_mesh['verts'], ctl_tris=ctl_mesh['faces'], tpl_verts=tpl_mesh['verts'], tpl_faces=tpl_mesh['faces'])
        deform.set_parameterization(ctl_tri_basis=ctl_tri_bs, vert_UVWs=vert_UVWs, vert_weights=vert_weights, vert_effect_idxs=vert_effect_idxs)

    mpaths = [(i, path)for i, path in enumerate(Path(M_DIR).glob('*.npy'))]
    n_files = len(mpaths)
    print(f'total files: {len(mpaths)}')
    pool = multiprocessing.Pool(processes=n_process)
    pool.map(partial(util_reconstruct_single_mesh, OUT_DIR=OUT_DIR, predictor=predictor, deformer=deform), mpaths)
    print('done')

    # for i, mdata_path in enumerate(Path(M_DIR).glob('*.npy')):
    #     # if 'CSR2776A' not in mdata_path.name:
    #     #     continue
    #     print(mdata_path)
    #
    #     # load 2d measurements
    #     mdata = np.load(mdata_path).item()
    #
    #     seg_dst_f = mdata['landmark_segment_dst_f']
    #     seg_dst_s = mdata['landmark_segment_dst_s']
    #     seg_locs_s = mdata['landmark_segment_location_s']
    #     seg_locs_f = mdata['landmark_segment_location_f']
    #     measurements = mdata['measurement']
    #     height = measurements['Height']
    #
    #     ctl_tri_mesh = predictor.predict(seg_dst_f, seg_dst_s, seg_locs_s, seg_locs_f, height)
    #
    #
    #     #out_path = f'{OUT_DIR}{mdata_path.stem}_ctl_tri.obj'
    #     #export_mesh(out_path, verts=ctl_tri_mesh['verts'], faces=ctl_tri_mesh['faces'])
    #     out_path = f'{OUT_DIR}{mdata_path.stem}_ctl_quad.obj'
    #     export_mesh(out_path, verts=ctl_tri_mesh['verts'], faces=ctl_mesh_quad_dom['faces'])
    #
    #     # tpl_new_verts, tpl_faces = deform.deform(ctl_tri_mesh['verts'])
    #     # out_path = f'{OUT_DIR}{mdata_path.stem}_tpl_deformed.obj'
    #     # export_mesh(out_path, verts=tpl_new_verts, faces=tpl_faces)
    #
    #     if i > 10:
    #         break


#caecar params
# -i
# /home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/vic_data.pkl
# -w
# /home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/global_parameterization.pkl
# -m
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_front_side_measurement
# -mo
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/models/fourier
# -o
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_front_side_ctl_meshes

#oh's mobile image params
#-i /home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/vic_data.pkl
# -w /home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/global_parameterization.pkl
# -m /home/khanhhh/data_1/projects/Oh/codes/body_measure/data/measurement/
# -mo /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/models/fourier
# -o ../data/ctr_mesh/