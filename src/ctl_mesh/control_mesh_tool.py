from ctl_mesh.control_mesh import  ControlMeshPredictor
from deformation.ffdt_deformation_lib import TemplateMeshDeform
from common.obj_util import export_mesh
import numpy as np
from pathlib import Path
import argparse
import pickle
import os
import multiprocessing
from functools import partial
import gc

g_debug_name = None

def util_reconstruct_single_mesh(record, OUT_DIR_CTL, OUT_DIR_DF, predictor, deformer):
    idx = record[0]
    mdata_path = record[1]

    #if idx % 100 == 0:
    #    print(f'{idx} - {mdata_path.name}')

    print(mdata_path.name)

    # load 2d measurements
    mdata = np.load(mdata_path).item()

    seg_dst_f = mdata['landmark_segment_dst_f']
    seg_dst_s = mdata['landmark_segment_dst_s']
    seg_locs_s = mdata['landmark_segment_location_s']
    seg_locs_f = mdata['landmark_segment_location_f']
    seg_f = mdata['landmark_segment_f']
    seg_s = mdata['landmark_segment_f']
    pose_joints_f = mdata['pose_joint_f']
    pose_joints_s = mdata['pose_joint_s']
    seg_s = mdata['landmark_segment_f']
    joint_3d_loc = mdata['joint_3D_location']
    measurements = mdata['measurement']
    height = measurements['Height']

    ctl_tri_mesh = predictor.predict(seg_dst_f, seg_dst_s, seg_locs_s, seg_locs_f, pose_joints_f, pose_joints_s, height)

    # out_path = f'{OUT_DIR}{mdata_path.stem}_ctl_tri.obj'
    # export_mesh(out_path, verts=ctl_tri_mesh['verts'], faces=ctl_tri_mesh['faces'])
    out_path = f'{OUT_DIR_CTL}{mdata_path.stem}_ctl_quad.obj'
    export_mesh(out_path, verts=ctl_tri_mesh['verts'], faces=ctl_tri_mesh['faces'])

    if deformer is not None:
        tpl_new_verts, tpl_faces = deform.deform(ctl_tri_mesh['verts'])
        out_path = f'{OUT_DIR_DF}{mdata_path.stem}_tpl_deformed.obj'
        export_mesh(out_path, verts=tpl_new_verts, faces=tpl_faces)

    gc.collect()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    ap.add_argument("-m", "--measure_dir", required=True, help="measurement 2d data directory")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for exporting control mesh slices")
    ap.add_argument("-w", "--weight", required=True, help="deform based on weight")
    ap.add_argument("-mo", "--model_dir", required=True, help="deform based on weight")
    ap.add_argument("-np", type=int, default = 1, help='')
    ap.add_argument("--deform", action='store_true', default = False, help='deformation or not')
    ap.add_argument("--nfiles", type=int, default = -1, help='just process this number of file')
    ap.add_argument("--debug_name", type=str, default = '', help='just process this number of file')

    args = ap.parse_args()

    in_path = args.input
    weight_path = args.weight
    M_DIR = args.measure_dir
    MODEL_DIR = args.model_dir
    OUT_DIR = args.out_dir + '/'
    OUT_DIR_CTL_MESH = f'{OUT_DIR}/caesar_mesh_control/'
    OUT_DIR_DF_MESH = f'{OUT_DIR}/caesar_mesh_deform/'
    os.makedirs(OUT_DIR_CTL_MESH, exist_ok=True)
    os.makedirs(OUT_DIR_DF_MESH, exist_ok=True)
    n_process = args.np

    g_debug_name = args.debug_name if args.debug_name != '' else None

    with open(in_path, 'rb') as f:
        data = pickle.load(f)

        ctl_mesh = data['control_mesh']
        ctl_mesh_quad_dom = data['control_mesh_quad_dom']
        slc_id_vert_idxs = data['slice_vert_idxs']
        slc_id_locs = data['slice_locs']
        ctl_sym_vert_pairs = data['control_mesh_symmetric_vert_pairs']
        mid_ankle_loc = np.array(data['mid_ankle_loc'])
        tpl_joint_locs = data['template_joint_locs']
        tpl_mesh = data['template_mesh']
        tpl_height = data['template_height']
        tpl_sym_vert_pairs = data['template_symmetric_vert_pairs']

        #arm_3d_tpl = data['arm_bone_locs']

        #load slice predictor
        predictor = ControlMeshPredictor(MODEL_DIR=MODEL_DIR)
        predictor.set_control_mesh(ctl_mesh=ctl_mesh, slc_id_vert_idxs=slc_id_vert_idxs, slc_id_locs=slc_id_locs, ctl_sym_vert_pairs=ctl_sym_vert_pairs, mid_ankle_loc=mid_ankle_loc)
        predictor.set_template_mesh(tpl_mesh=tpl_mesh, tpl_height=tpl_height, tpl_joint_locs=tpl_joint_locs)

    #load deform
    if args.deform is True:
        with open(weight_path, 'rb') as f:
            data = pickle.load(f)
            ctl_tri_bs = data['control_mesh_tri_basis']
            vert_UVWs = data['template_vert_UVW']
            vert_weights = data['template_vert_weight']
            vert_effect_idxs = data['template_vert_effect_idxs']

            debug = False
            if debug:
                debug_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_limb_faces.pkl'
                with open(debug_path, 'rb') as file:
                    tpl_limb_faces = pickle.load(file)
                    tpl_limb_faces = set(tpl_limb_faces)
                    tpl_faces_org = tpl_mesh['faces']
                    tpl_faces = []
                    for idx, face in enumerate(tpl_faces_org):
                        if idx not in tpl_limb_faces:
                            tpl_faces.append(face)
            else:
                tpl_faces = tpl_mesh['faces']

            deform = TemplateMeshDeform(effective_range=4, use_mean_rad=False)
            deform.set_meshes(ctl_verts=ctl_mesh['verts'], ctl_tris=ctl_mesh['faces'], tpl_verts=tpl_mesh['verts'], tpl_faces=tpl_faces)
            deform.set_parameterization(ctl_tri_basis=ctl_tri_bs, vert_tri_UVWs=vert_UVWs, vert_tri_weights=vert_weights, vert_effect_tri_idxs=vert_effect_idxs)
    else:
        deform = None

    gc.collect()

    mpaths = [(i, path)for i, path in enumerate(Path(M_DIR).glob('*.npy'))]
    paths_names = [record[1].stem for record in mpaths]
    sorted_idxs = sorted(range(len(paths_names)),key=paths_names.__getitem__)

    nfiles = len(mpaths) if args.nfiles <= 0 else args.nfiles
    process_idxs = sorted_idxs[:nfiles]
    mpaths = [mpaths[idx] for idx in process_idxs]

    if args.debug_name != '':
        debug_path = None
        for idx, path in mpaths:
            if args.debug_name in path.stem:
                debug_path = path
                break
        assert debug_path is not None
        mpaths = [(0, debug_path)]

    print(f'total files: {len(mpaths)}')
    pool = multiprocessing.Pool(processes=n_process)
    pool.map(partial(util_reconstruct_single_mesh, OUT_DIR_CTL=OUT_DIR_CTL_MESH, OUT_DIR_DF = OUT_DIR_DF_MESH, predictor=predictor, deformer=deform), mpaths)
    print('done')


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