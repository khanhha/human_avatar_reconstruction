import numpy as np
import argparse
import os
from pathlib import Path
import pickle
from src.obj_util import export_mesh
import src.util as util

def define_id_mapping():
    mappings = {}
    mappings['L2_RKnee'] = 'Knee'
    mappings['L2_LKnee'] = 'Knee'
    mappings['L9_Aux_Waist_UnderBust'] = 'Aux_Waist_UnderBust_0'
    mappings['L6_Hip'] = 'Hip'
    mappings['L3_RMidThigh'] = 'Aux_Thigh_0'
    mappings['L3_LMidThigh'] = 'Aux_Thigh_0'
    mappings['L7_Aux_Hip_Waist'] = 'Aux_Hip_Waist_0'
    mappings['L8_Waist'] = 'Waist'
    mappings['L12_Armcye'] = 'Armscye'
    mappings['L14_Shoulder'] = 'Shoulder'
    mappings['L0_LAnkle'] = 'Ankle'
    mappings['L0_RAnkle'] = 'Ankle'
    mappings['L1_RCalf'] = 'Calf'
    mappings['L1_LCalf'] = 'Calf'
    mappings['L11_Bust'] = 'Bust'
    mappings['L15_Collar'] = 'Collar'
    mappings['L13_Aux_Armcye_Shoulder'] = 'Aux_Armscye_Shoulder_0'
    mappings['L10_UnderBust'] = 'UnderBust'
    mappings['L4_Crotch'] = 'Crotch'
    mappings['L16_Neck'] = 'Neck'
    mappings['L5_Aux_Crotch_Hip'] = 'Aux_Crotch_Hip_0'

    return mappings

def deform_slice(slice, w, d, z = -1, slice_org = None):
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
    if z != -1:
        nslice[:,2]  = z
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vic", "--victoria", required=True, help="victoria pkl file")
    ap.add_argument("-m", "--measure_dir", required=True, help="measurement 2d data directory")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    args = vars(ap.parse_args())

    vic_path = args['victoria']
    M_DIR = args['measure_dir']
    OUT_DIR = args['out_dir'] + '/'

    for fpath in Path(OUT_DIR).glob('*.*'):
        os.remove(fpath)

    slice_locs = None
    vic_height = None
    with open(vic_path, 'rb') as f:
        data = pickle.load(f)
        ctl_mesh = data['control_mesh']
        slc_id_vert_idxs = data['slice_vert_idxs']
        slc_id_locs = data['slice_locs']
        ctl_tri_bs = data['control_mesh_tri_basis']

        tpl_mesh = data['template_mesh']
        vic_height = data['template_height']
        vert_UVWs = data['template_vert_UVW']
        vert_weights = data['template_vert_weight']
        vert_effect_idxs = data['template_vert_effect_idxs']
        print(len(vert_weights[0]))
        print(len(vert_effect_idxs[0]))
        print(len(vert_UVWs[0]))

    print('control  mesh: nverts = {0}, ntris = {1}'.format(ctl_mesh['verts'].shape[0], len(ctl_mesh['faces'])))
    print('victoria mesh: nverts = {0}, ntris = {1}'.format(tpl_mesh['verts'].shape[0], len(tpl_mesh['faces'])))


    for mdata_path in Path(M_DIR).glob('*.npy'):
        print(mdata_path)
        mdata = np.load(mdata_path )
        seg_dst_f = mdata.item().get('landmark_segment_dst_f')
        seg_dst_s = mdata.item().get('landmark_segment_dst_s')
        seg_height_locs = mdata.item().get('landmark_segment_height')
        measurements = mdata.item().get('measurement')
        height = measurements['Height']

        h_ratio = vic_height/height

        id_mappings = define_id_mapping()
        ct_mesh_slices = {}

        ctl_new_mesh = deepcopy(ctl_mesh)

        for id_3d, id_2d in id_mappings.items():
            #debug
            #if id_3d not in ['L0_RAnkle', 'L0_LAnkle']:
            #    continue

            slc_idxs = slc_id_vert_idxs[id_3d]
            slice = ctl_mesh['verts'][slc_idxs]

            if id_2d in seg_dst_f:
                w = seg_dst_f[id_2d]
                d = w
                if id_2d in seg_dst_s:
                    d = seg_dst_s[id_2d]
                z = -1
                if id_2d in seg_height_locs:
                    z = seg_height_locs[id_2d]

                if z == -1:
                    print(f'z of {id_2d} is not available. ignore this slice')
                    continue

                #transform to victoria's scale
                w = w*h_ratio
                d = d*h_ratio
                z = z*h_ratio

                #print('slice = {0:25}, width = {1:20}, depth = {2:20}, height = {3:20}'.format(id_2d, w, d, z))
                slc_org = slc_id_locs[id_3d]
                slice_out = deform_slice(slice, w, d, z, slice_org = slc_org)
                ctl_new_mesh['verts'][slc_idxs, :] = slice_out
            else:
                print(f'missing measurement {id_2d}')

        ctl_df_basis = util.calc_triangle_local_basis(ctl_new_mesh['verts'], ctl_new_mesh['faces'])
        tpl_df_mesh = deform_template_mesh(tpl_mesh, vert_effect_idxs, vert_weights, vert_UVWs, ctl_df_basis)

        out_path = f'{OUT_DIR}{mdata_path.stem}_ctl.obj'
        export_mesh(out_path, ctl_new_mesh['verts'], ctl_new_mesh['faces'])

        out_path = f'{OUT_DIR}{mdata_path.stem}_tpl.obj'
        export_mesh(out_path, tpl_mesh['verts'], tpl_mesh['faces'])

        out_path = f'{OUT_DIR}{mdata_path.stem}_deform.obj'
        export_mesh(out_path, tpl_df_mesh['verts'], tpl_df_mesh['faces'])


