import numpy as np
import argparse
import os
from pathlib import Path
import pickle
from numpy.linalg import norm
from src.obj_util import export_mesh
import src.util as util

def define_slice_id_mapping():
    mappings = {}
    mappings['Slice_LAnkle'] = 'Ankle'
    mappings['Slice_RAnkle'] = 'Ankle'
    mappings['Slice_RCalf'] = 'Calf'
    mappings['Slice_LCalf'] = 'Calf'
    mappings['Slice_RKnee'] = 'Knee'
    mappings['Slice_LKnee'] = 'Knee'
    mappings['Slice_LKnee_Crotch_0'] = 'Aux_Knee_Crotch_0'
    mappings['Slice_LKnee_Crotch_1'] = 'Aux_Knee_Crotch_1'
    mappings['Slice_RKnee_Crotch_0'] = 'Aux_Knee_Crotch_0'
    mappings['Slice_RKnee_Crotch_1'] = 'Aux_Knee_Crotch_1'
    mappings['Slice_Crotch'] = 'Crotch'
    mappings['Slice_Crotch_Hip_0'] = 'Aux_Crotch_Hip_0'
    mappings['Slice_Hip'] = 'Hip'
    mappings['Slice_Hip_Waist_0'] = 'Aux_Hip_Waist_0'
    mappings['Slice_Waist'] = 'Waist'
    mappings['Slice_Waist_UnderBust_0'] = 'Aux_Waist_UnderBust_0'
    mappings['Slice_UnderBust'] = 'UnderBust'
    mappings['Slice_Bust'] = 'Bust'
    mappings['Slice_Armscye'] = 'Armscye'
    mappings['Slice_Armscye_Shoulder_0'] = 'Aux_Armscye_Shoulder_0'
    mappings['Slice_Shoulder'] = 'Shoulder'
    mappings['Slice_Collar'] = 'Collar'
    mappings['Slice_Neck'] = 'Neck'
    return mappings

def bone_lengths(arm_2d, ratio):
    lengths = {}
    lengths['Shin'] = ratio*norm(arm_2d['LAnkle'] - arm_2d['LKnee'])
    lengths['Thigh'] = ratio*norm(arm_2d['LHip'] - arm_2d['LKnee'])
    lengths['Torso'] = ratio*norm(arm_2d['MidHip'] - arm_2d['Neck'])
    lengths['Shoulder'] = ratio*norm(arm_2d['LShoulder'] - arm_2d['Neck'])
    lengths['UpperArm'] = ratio*norm(arm_2d['LShoulder'] - arm_2d['LElbow'])
    lengths['ForeArm'] = ratio*norm(arm_2d['LElbow'] - arm_2d['LWrist'])
    return lengths

def scale_tpl_armature(arm_3d, arm_2d, ratio):
    #leg
    blengths = bone_lengths(arm_2d, ratio)
    arm_3d['LKnee'] = arm_3d['LAnkle'] + blengths['Shin']*util.normalize(arm_3d['LKnee'] - arm_3d['LAnkle'])
    arm_3d['LHip']  = arm_3d['LKnee']  + blengths['Thigh']*util.normalize(arm_3d['LHip'] - arm_3d['LKnee'])
    arm_3d['RKnee'] = arm_3d['RAnkle'] + blengths['Shin']*util.normalize(arm_3d['RKnee'] - arm_3d['RAnkle'])
    arm_3d['RHip']  = arm_3d['RKnee']  + blengths['Thigh']*util.normalize(arm_3d['RHip'] - arm_3d['RKnee'])

    midhip = 0.5*(arm_3d['LHip'] + arm_3d['RHip'])
    arm_3d['Neck'] = midhip + blengths['Torso']*util.normalize(arm_3d['Neck'] - midhip)
    #TODO: Crotch, Spine, Chest

    arm_3d['LShoulder'] = arm_3d['Neck'] + blengths['Shoulder']*util.normalize(arm_3d['LShoulder'] - arm_3d['Neck'])
    arm_3d['RShoulder'] = arm_3d['Neck'] + blengths['Shoulder']*util.normalize(arm_3d['RShoulder'] - arm_3d['Neck'])

    arm_3d['LElbow'] = arm_3d['LShoulder'] + blengths['UpperArm']*util.normalize(arm_3d['LElbow'] - arm_3d['LShoulder'])
    arm_3d['RElbow'] = arm_3d['RShoulder'] + blengths['UpperArm']*util.normalize(arm_3d['RElbow'] - arm_3d['RShoulder'])

    arm_3d['LWrist'] = arm_3d['LElbow'] + blengths['ForeArm']*util.normalize(arm_3d['LWrist'] - arm_3d['LElbow'])
    arm_3d['RWrist'] = arm_3d['RElbow'] + blengths['ForeArm']*util.normalize(arm_3d['RWrist'] - arm_3d['RElbow'])

    return arm_3d

#the basis origin of armature is the midhip position
def transform_non_vertical_slice(slice, loc, loc_after, radius):
    slice_n = slice - loc
    rads = norm(slice_n, axis=1)
    max_rad = np.max(rads)
    scale = radius / max_rad
    slice_n *= scale
    return loc_after + slice_n

def transform_vertical_slice(slice, w, d, slice_org = None):
    if slice_org is None:
        slice_org = np.mean(slice, axis=0)
    nslice = slice - slice_org
    range = np.max(nslice, axis=0) - np.min(nslice, axis=0)
    w_ratio = w / range[0]
    d_ratio = d / range[1]
    #print(w_ratio, d_ratio)
    nslice[:,0] *= w_ratio
    nslice[:,1] *= d_ratio
    nslice = nslice + slice_org
    return nslice

from copy import deepcopy
def deform_template_mesh(org_mesh, effect_vert_tri_idxs, vert_weights, vert_UVWs, ctl_df_basis):
    df_mesh  = deepcopy(org_mesh)
    df_verts = df_mesh['verts']
    for i in range(df_verts.shape[0]):
        df_co = np.zeros(3, np.float32)
        W = 0.0
        for idx, ev_idx in enumerate(effect_vert_tri_idxs[i]):
            df_basis = ctl_df_basis[ev_idx,:,:]
            uvw = vert_UVWs[i][idx]
            co = df_basis[0,:] + uvw[0]*df_basis[1,:]+ uvw[1]*df_basis[2,:]+ uvw[2]*df_basis[3,:]
            w_tri = vert_weights[i][idx]
            df_co += w_tri * co
            W += w_tri
        if W > 0:
            df_co /= W
            df_verts[i,:] = df_co
    return df_mesh

def transform_arm_slices(mesh, slc_id_locs, slc_id_vert_idxs, arm_3d):
    slc_org = slc_id_locs['Slice_LElbow']
    slc_idxs = slc_id_vert_idxs['Slice_LElbow']
    slice = ctl_mesh['verts'][slc_idxs]
    radius = h_ratio * 0.5 * seg_dst_f['Elbow']
    mesh['verts'][slc_idxs, :] = transform_non_vertical_slice(slice, slc_org, arm_3d['LElbow'], radius)

    slc_org = slc_id_locs['Slice_LWrist']
    slc_idxs = slc_id_vert_idxs['Slice_LWrist']
    slice = ctl_mesh['verts'][slc_idxs]
    wrist_radius = h_ratio * 0.5 * 0.8 * seg_dst_f['Elbow']
    mesh['verts'][slc_idxs, :] = transform_non_vertical_slice(slice, slc_org, arm_3d['LWrist'], wrist_radius)

    slc_org = slc_id_locs['Slice_RElbow']
    slc_idxs = slc_id_vert_idxs['Slice_RElbow']
    slice = ctl_mesh['verts'][slc_idxs]
    radius = h_ratio * 0.5 * seg_dst_f['Elbow']
    mesh['verts'][slc_idxs, :] = transform_non_vertical_slice(slice, slc_org, arm_3d['RElbow'], radius)

    slc_org = slc_id_locs['Slice_LWrist']
    slc_idxs = slc_id_vert_idxs['Slice_LWrist']
    slice = ctl_mesh['verts'][slc_idxs]
    wrist_radius = h_ratio * 0.5 * 0.8 * seg_dst_f['Elbow']
    mesh['verts'][slc_idxs, :] = transform_non_vertical_slice(slice, slc_org, arm_3d['LWrist'], wrist_radius)

    lhand_idxs = []
    for id, idxs in slc_id_vert_idxs.items():
        if 'LHand' in id:
            lhand_idxs.append(idxs[:])
    displacement = arm_3d['LWrist'] - slc_id_locs['Slice_LWrist']
    mesh['verts'][lhand_idxs, :] += displacement

    rhand_idxs = []
    for id, idxs in slc_id_vert_idxs.items():
         if 'RHand' in id:
             rhand_idxs.append(idxs[:])
    displacement = arm_3d['RWrist'] - slc_id_locs['Slice_RWrist']
    mesh['verts'][rhand_idxs, :] += displacement

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    ap.add_argument("-m", "--measure_dir", required=True, help="measurement 2d data directory")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    args = vars(ap.parse_args())

    IN_DIR = args['input']
    M_DIR = args['measure_dir']
    OUT_DIR = args['out_dir'] + '/'

    for fpath in Path(OUT_DIR).glob('*.*'):
        os.remove(fpath)

    slice_locs = None
    vic_height = None
    with open(f'{IN_DIR}/vic_data.pkl', 'rb') as f:
        data = pickle.load(f)
        ctl_mesh = data['control_mesh']
        slc_id_vert_idxs = data['slice_vert_idxs']
        slc_id_locs = data['slice_locs']
        ctl_tri_bs = data['control_mesh_tri_basis']
        arm_3d = data['arm_bone_locs']
        tpl_mesh = data['template_mesh']
        vic_height = data['template_height']

    with open(f'{IN_DIR}/vic_weight.pkl', 'rb') as f:
        data = pickle.load(f)
        vert_UVWs = data['template_vert_UVW']
        vert_weights = data['template_vert_weight']
        vert_effect_idxs = data['template_vert_effect_idxs']

    print('control  mesh: nverts = {0}, ntris = {1}'.format(ctl_mesh['verts'].shape[0], len(ctl_mesh['faces'])))
    print('victoria mesh: nverts = {0}, ntris = {1}'.format(tpl_mesh['verts'].shape[0], len(tpl_mesh['faces'])))

    out_path = f'{OUT_DIR}victoria_ctl.obj'
    export_mesh(out_path, ctl_mesh['verts'], ctl_mesh['faces'])

    out_path = f'{OUT_DIR}victoria_tpl.obj'
    export_mesh(out_path, tpl_mesh['verts'], tpl_mesh['faces'])

    for mdata_path in Path(M_DIR).glob('*.npy'):
        print(mdata_path)
        mdata = np.load(mdata_path )
        seg_dst_f = mdata.item().get('landmark_segment_dst_f')
        seg_dst_s = mdata.item().get('landmark_segment_dst_s')
        seg_location = mdata.item().get('landmark_segment_location_s')
        measurements = mdata.item().get('measurement')
        height = measurements['Height']

        arm_2d_f = mdata.item().get('armature_f')
        h_ratio = vic_height/height
        arm_3d = scale_tpl_armature(arm_3d, arm_2d_f, h_ratio)

        id_mappings = define_slice_id_mapping()
        ct_mesh_slices = {}

        ctl_new_mesh = deepcopy(ctl_mesh)

        #hack: the background z value extracted from image is not exact. therefore, we consider ankle z as the z starting point
        tpl_ankle_hor = slc_id_locs['Slice_LAnkle'][1]
        tpl_ankle_ver = slc_id_locs['Slice_LAnkle'][2]
        #slice location in relative to ankle location in side image
        for id_3d, id_2d in id_mappings.items():
            #debug
            #if id_3d not in ['L0_RAnkle', 'L0_LAnkle']:
            #    continue
            if id_3d  not in slc_id_vert_idxs:
                print(f'indices of {id_3d} are not available')
                continue

            slc_idxs = slc_id_vert_idxs[id_3d]
            slice = ctl_mesh['verts'][slc_idxs]

            if id_2d in seg_dst_f:
                w = seg_dst_f[id_2d]
                d = w
                if id_2d in seg_dst_s:
                    d = seg_dst_s[id_2d]

                seg_log = None
                if id_2d in seg_location:
                    seg_log = seg_location[id_2d]

                if seg_log is None:
                    print(f'location of {id_2d} is not available. ignore this slice')
                    continue

                slc_loc_hor = seg_log[0]
                slc_loc_ver = np.abs(seg_log[1])

                #transform to victoria's scale
                w = w*h_ratio
                d = d*h_ratio
                slc_loc_ver = slc_loc_ver * h_ratio
                slc_loc_hor = slc_loc_hor * h_ratio

                print('slice = {0:25}, width = {1:20}, depth = {2:20}, hor = {3:20}, ver = {4:20}'.format(id_2d, w, d, slc_loc_hor, slc_loc_ver))

                #print('slice = {0:25}, width = {1:20}, depth = {2:20}, height = {3:20}'.format(id_2d, w, d, z))
                slc_org = slc_id_locs[id_3d]
                slice_out = transform_vertical_slice(slice, w, d, slice_org = slc_org)

                slice_out[:, 2] = slc_loc_ver + tpl_ankle_ver

                slice_hor_center = 0.5*(np.min(slice_out[:,1]) + np.max(slice_out[:,1]))
                slice_out[:, 1] += (-slice_hor_center + tpl_ankle_hor + slc_loc_hor)

                ctl_new_mesh['verts'][slc_idxs, :] = slice_out
            else:
                print(f'missing measurement {id_2d}')

        #transform_arm_slices(ctl_new_mesh, slc_id_locs, slc_id_vert_idxs, arm_3d)

        ctl_df_basis = util.calc_triangle_local_basis(ctl_new_mesh['verts'], ctl_new_mesh['faces'])
        #tpl_df_mesh = deform_template_mesh(tpl_mesh, vert_effect_idxs, vert_weights, vert_UVWs, ctl_df_basis)

        out_path = f'{OUT_DIR}{mdata_path.stem}_ctl.obj'
        export_mesh(out_path, ctl_new_mesh['verts'], ctl_new_mesh['faces'])

        out_path = f'{OUT_DIR}{mdata_path.stem}_deform.obj'
        #export_mesh(out_path, tpl_df_mesh['verts'], tpl_df_mesh['faces'])