import numpy as np
import argparse
import os
from pathlib import Path
from src.obj_util import  load_slice_template_from_obj_file, export_slice_obj

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



def deform_slice(slice, w, d, h = -1):
    mean = np.mean(slice, axis=0)
    nslice = slice - mean
    z_min = np.min(nslice[:,2])
    z_max = np.max(nslice[:,2])
    range = np.max(nslice, axis=0) - np.min(nslice, axis=0)
    w_ratio = w / range[0]
    d_ratio = d / range[1]
    #print(w_ratio, d_ratio)
    nslice[:,0] *= w_ratio
    nslice[:,1] *= d_ratio
    if h != -1:
        nslice[:,2]  = h
    return nslice + mean

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--3D_template", required=True, help="slice obj directory")
    ap.add_argument("-m", "--measure_data", required=True, help="measurement 2d data")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    args = vars(ap.parse_args())

    SLICE_DIR = args['3D_template'] + '/'
    mdata_path = args['measure_data']
    OUT_DIR = args['out_dir'] + '/'

    for fpath in Path(OUT_DIR).glob('*.*'):
        os.remove(fpath)

    slices = {}
    for fpath in Path(SLICE_DIR).glob('*.obj'):
        slice = load_slice_template_from_obj_file(fpath)
        slice = hack_fix_noise_vertices(fpath, slice)
        export_slice_obj(fpath, slice)
        slices[fpath.stem] = slice

    mdata = np.load(mdata_path )
    seg_dst_f = mdata.item().get('landmark_segment_dst_f')
    seg_dst_s = mdata.item().get('landmark_segment_dst_s')
    seg_height = mdata.item().get('landmark_segment_height')

    id_mappings = define_id_mapping()
    ct_mesh_slices = {}
    for id_3d, id_2d in id_mappings.items():
        #debug
        #if id_3d not in ['L0_RAnkle', 'L0_LAnkle']:
        #    continue

        slice = slices[id_3d]
        if id_2d in seg_dst_f:

            w = seg_dst_f[id_2d]
            d = w
            if id_2d in seg_dst_s:
                d = seg_dst_s[id_2d]
            h = -1
            if id_2d in seg_height:
                h = seg_height[id_2d]
                #h = h /70

            #w, d = w/70, d/70
            #print(w,d)
            print('slice = {0:25}, width = {1:20}, depth = {2:20}, height = {3:20}'.format(id_2d, w, d, h))
            sl = deform_slice(slice, w, d, h)
            ct_mesh_slices[id_3d] = sl
        else:
            print(f'missing measurement {id_2d}')

    for id_3d, slice in ct_mesh_slices.items():
        fpath = f'{OUT_DIR}{id_3d}.obj'
        export_slice_obj(fpath, slice)

