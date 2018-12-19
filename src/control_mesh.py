import numpy as np
import argparse
import os
from pathlib import Path
import pickle
from numpy.linalg import norm
from src.obj_util import export_mesh
import src.util as util
import shapely.geometry as geo
from shapely.ops import nearest_points
from src.caesar_rbf_net import RBFNet
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, KDTree
from copy import copy

def slice_id_3d_2d_mappings():
    mappings = {}
    mappings['LAnkle'] = 'Ankle'
    mappings['RAnkle'] = 'Ankle'

    mappings['RCalf'] = 'Calf'
    mappings['LCalf'] = 'Calf'

    mappings['RKnee'] = 'Knee'
    mappings['LKnee'] = 'Knee'

    mappings['LAux_Knee_UnderCrotch_0'] = 'Aux_Knee_UnderCrotch_0'
    mappings['LAux_Knee_UnderCrotch_1'] = 'Aux_Knee_UnderCrotch_1'
    mappings['LAux_Knee_UnderCrotch_2'] = 'Aux_Knee_UnderCrotch_2'
    mappings['LAux_Knee_UnderCrotch_3'] = 'Aux_Knee_UnderCrotch_3'
    mappings['RAux_Knee_UnderCrotch_0'] = 'Aux_Knee_UnderCrotch_0'
    mappings['RAux_Knee_UnderCrotch_1'] = 'Aux_Knee_UnderCrotch_1'
    mappings['RAux_Knee_UnderCrotch_2'] = 'Aux_Knee_UnderCrotch_2'
    mappings['RAux_Knee_UnderCrotch_3'] = 'Aux_Knee_UnderCrotch_3'

    mappings['RUnderCrotch'] = 'UnderCrotch'
    mappings['LUnderCrotch'] = 'UnderCrotch'

    mappings['Crotch'] = 'Crotch'
    mappings['Aux_Crotch_Hip_0'] = 'Aux_Crotch_Hip_0'
    mappings['Aux_Crotch_Hip_1'] = 'Aux_Crotch_Hip_1'
    mappings['Aux_Crotch_Hip_2'] = 'Aux_Crotch_Hip_2'

    mappings['Hip'] = 'Hip'
    mappings['Aux_Hip_Waist_0'] = 'Aux_Hip_Waist_0'
    mappings['Waist'] = 'Waist'
    mappings['Aux_Waist_UnderBust_0'] = 'Aux_Waist_UnderBust_0'
    mappings['Aux_Waist_UnderBust_1'] = 'Aux_Waist_UnderBust_1'
    mappings['UnderBust'] = 'UnderBust'
    mappings['Aux_UnderBust_Bust_0'] = 'Aux_UnderBust_Bust_0'
    mappings['Bust'] = 'Bust'
    mappings['Armscye'] = 'Armscye'
    mappings['Aux_Armscye_Shoulder_0'] = 'Aux_Armscye_Shoulder_0'
    mappings['Shoulder'] = 'Shoulder'
    mappings['Collar'] = 'Collar'
    mappings['Neck'] = 'Neck'
    return mappings

def breast_part_slice_id_3d_2d_mappings():
    id_2ds = ['Breast_Depth_Aux_UnderBust_Bust_0', 'Breast_Depth_Bust', 'Breast_Depth_Aux_Bust_Armscye_0']
    mappings = {}
    for id_2d in id_2ds:
        mappings['R' + id_2d] = id_2d

    for id_2d in id_2ds:
        mappings['L' + id_2d] = id_2d
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

def subdivide_catmull_temp(mesh):
    verts = mesh['verts']
    faces = mesh['faces']
    edges = mesh['edges']
    loops = mesh['loops'] #l[0]: vertex, l[1]: edge

    nv = len(verts)
    nf = len(faces)
    ne = len(edges)

    e_f = mesh['e_f']
    v_f = mesh['v_f']
    v_e = mesh['v_e']
    f_l = mesh['f_l']

    new_vert_pnts = np.zeros(shape=(nv,3), dtype=np.float32)
    new_edge_pnts = np.zeros(shape=(ne,3), dtype=np.float32)
    new_face_pnts = np.zeros(shape=(nf,3), dtype=np.float32)

    for f_idx in range(nf):
        co = np.zeros(shape=(3,), dtype=np.float32)
        f = faces[f_idx]
        for vf_idx in f:
            co += verts[vf_idx, :]
        co /= len(f)
        new_face_pnts[f_idx, :] = co

    for e_idx in range(ne):
        co = np.zeros(shape=(3,))
        for ef_idx in e_f[e_idx]:
            co += new_face_pnts[ef_idx,:]

        ev0 = edges[e_idx][0]
        ev1 = edges[e_idx][1]
        co += verts[ev0,:]
        co += verts[ev1,:]
        new_edge_pnts[e_idx, :] = co/(2.0 + len(e_f[e_idx]))

    for v_idx in range(nv):
        co_avg_f = np.zeros(shape=(3,))
        co_avg_e = np.zeros(shape=(3,))

        nvf = len(v_f[v_idx])
        for vf_idx in v_f[v_idx]:
            co_avg_f += new_face_pnts[vf_idx, :]
        co_avg_f /= nvf

        nve = len(v_e[v_idx])
        for ve_idx in v_e[v_idx]:
            co_avg_e += new_edge_pnts[ve_idx, :]
        co_avg_e /= nve

        #not a boundary vertex
        if nve == nvf:
            new_vert_pnts[v_idx, :] = (co_avg_f + 2.0*co_avg_e + (nve - 3.0) * verts[v_idx, :]) / nve
        else:
            new_vert_pnts[v_idx, :] = verts[v_idx,:]

    #create new topology
    new_vert_pnts = np.concatenate([new_vert_pnts, new_face_pnts, new_edge_pnts], axis=0)
    new_faces = []
    for f_idx in range(nf):
        nl = len(f_l[f_idx])
        for idx in range(nl):
            l_0 = loops[f_l[f_idx][idx]]
            l_1 = loops[f_l[f_idx][(idx + 1) % nl]]
            v0 = nv + nf + l_0[1]
            v1 = l_1[0]
            v2 = nv + nf + l_1[1]
            v3 = nv + f_idx
            new_faces.append((v0, v1, v2, v3))

    new_mesh = {'verts':new_vert_pnts, 'faces': new_faces}

    return new_mesh

def projec_mesh_onto_mesh(mesh_0, mesh_0_vert_idxs, mesh_1):
    verts_0 = mesh_0['verts']
    nv_0 = len(verts_0)

    faces_1 = mesh_1['faces']
    verts_1 = mesh_1['verts']
    nf_1 = len(faces_1)

    centroid_faces_1 = np.zeros(shape=(nf_1, 3), dtype=np.float)
    for f_idx in range(nf_1):

        co = np.zeros(shape=(3,), dtype=np.float32)
        for v_idx in faces_1[f_idx]:
            co += verts_1[v_idx,:]
        co /= len(faces_1[f_idx])

        centroid_faces_1[f_idx, :] = co

    # out_path = f'{OUT_DIR}{mdata_path.stem}_debug_centroid_faces.obj'
    # print(f'\toutput deformed mesh to {out_path}')
    # export_mesh(out_path, centroid_faces_1, [])

    # out_path = f'{OUT_DIR}{mdata_path.stem}_debug_mesh_0.obj'
    # print(f'\toutput deformed mesh to {out_path}')
    # export_mesh(out_path, verts_0, [])

#    tree = cKDTree(centroid_faces_1, leafsize=5)
    tree = KDTree(centroid_faces_1, leafsize=5)

    if mesh_0_vert_idxs is None:
        mesh_0_vert_idxs = range(nv_0)

    for v_idx in mesh_0_vert_idxs:
        v_co = verts_0[v_idx,:]
        #query closest faces
        _, idxs = tree.query(v_co, 5)
        cls_dst = np.inf
        cls_pnt = v_co
        for f_idx in idxs:
            f = faces_1[f_idx]
            assert len(f) == 4
            on_quad_p = util.closest_on_quad_to_point_v3(v_co, verts_1[f[0],:], verts_1[f[1],:], verts_1[f[2],:], verts_1[f[3],:])
            dst = norm(on_quad_p - v_co)
            if dst < cls_dst:
                cls_dst = dst
                #cls_pnt = 0.25*(verts_1[f[0],:] + verts_1[f[1],:] + verts_1[f[2],:] + verts_1[f[3],:])
                cls_pnt = on_quad_p
        verts_0[v_idx, :] = cls_pnt

    #out_path = f'{OUT_DIR}{mdata_path.stem}_debug_projected.obj'
    #print(f'\toutput deformed mesh to {out_path}')
    #export_mesh(out_path, np.array(test_points), [])

#the basis origin of armature is the midhip position
def transform_non_vertical_slice(slice, loc, loc_after, radius):
    slice_n = slice - loc
    rads = norm(slice_n, axis=1)
    max_rad = np.max(rads)
    scale = radius / max_rad
    slice_n *= scale
    return loc_after + slice_n

def scale_vertical_slice(slice, w_ratio, d_ratio, scale_center = None):
    if scale_center is None:
        scale_center = np.mean(slice, axis=0)
    nslice = slice - scale_center
    nslice[:,0] *= w_ratio
    nslice[:,1] *= d_ratio
    nslice = nslice + scale_center
    return nslice

from copy import deepcopy
def deform_template_mesh(org_mesh, effect_vert_tri_idxs, vert_weights, vert_UVWs, ctl_df_basis, deform_vert_idxs = None):
    df_mesh  = deepcopy(org_mesh)
    df_verts = df_mesh['verts']
    nv = len(df_verts)

    if deform_vert_idxs == None:
        deform_vert_idxs = range(nv)

    for i in deform_vert_idxs:
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

def is_breast_segment(id_seg_2d):
    if id_seg_2d == 'Aux_UnderBust_Bust_0' or \
        id_seg_2d == 'Bust' or \
        id_seg_2d == 'Aux_Bust_Armscye_0':
        return True
    else:
        return False

#note: this function just works for breast slice
def scale_breast_height(brst_slc, slc_height):
    #hack
    #find two ending points of the breast slice
    #1. the point with smallest x
    end_0_idx = np.argmin(brst_slc[:,0])
    #2. the point with lagrest y
    end_1_idx = np.argmax(brst_slc[:,1])
    brst_hor_seg = geo.LineString([brst_slc[end_0_idx,:2], brst_slc[end_1_idx,:2]])

    #3: breast highest height
    cur_brst_height = 0.0
    for i in range(brst_slc.shape[0]):
        if i != end_0_idx and i != end_1_idx:
            p = brst_slc[i,:2]
            dst = geo.Point(p).distance(brst_hor_seg)
            cur_brst_height = max(dst, cur_brst_height)

    ratio =  slc_height/cur_brst_height

    for i in range(brst_slc.shape[0]):
        if i != end_0_idx and i != end_1_idx:
            p = brst_slc[i,:2]
            proj = nearest_points(brst_hor_seg, geo.Point(p))[0]
            proj = np.array(proj).flatten()
            p_scaled = proj + ratio*(p-proj)
            brst_slc[i,:2] = p_scaled

    brst_slc[end_0_idx, 2] = brst_slc[end_0_idx, 2] + 0.01
    brst_slc[end_1_idx, 2] = brst_slc[end_1_idx, 2] - 0.01

    return brst_slc, end_0_idx, end_1_idx

def fix_crotch_hip_cleavage(slice, depth):
    delta = 0.05*depth
    clv_idx = 0
    slice[clv_idx, 1] = slice[clv_idx, 1] - delta
    return slice

import sys
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    ap.add_argument("-m", "--measure_dir", required=True, help="measurement 2d data directory")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    ap.add_argument("-w", "--weight", required=True, help="deform based on weight")
    ap.add_argument("-mo", "--model_dir", required=True, help="deform based on weight")

    args = vars(ap.parse_args())

    in_path = args['input']
    M_DIR = args['measure_dir']
    OUT_DIR = args['out_dir'] + '/'

    weight_path = args['weight']

    MODEL_DIR = args['model_dir']
    models = {}
    for path in Path(MODEL_DIR).glob('*.pkl'):
        models[path.stem] = RBFNet.load_from_path(path)
    print('load models: ', models.keys())

    for fpath in Path(OUT_DIR).glob('*.*'):
        os.remove(str(fpath))

    with open(in_path, 'rb') as f:
        data = pickle.load(f)
        ctl_mesh = data['control_mesh']
        ctl_mesh_quad_dom = data['control_mesh_quad_dom']
        slc_id_vert_idxs = data['slice_vert_idxs']
        slc_id_locs = data['slice_locs']
        arm_3d_tpl = data['arm_bone_locs']

        ctl_sym_vert_pairs = data['control_mesh_symmetric_vert_pairs']

        tpl_sym_vert_pairs = data['template_symmetric_vert_pairs']
        tpl_lbody_vert_idxs = [pair[0] for pair in tpl_sym_vert_pairs]
        tpl_rbody_vert_idxs = [pair[1] for pair in tpl_sym_vert_pairs]

        tpl_mesh = data['template_mesh']
        tpl_height = data['template_height']

    print('control  mesh: nverts = {0}, ntris = {1}'.format(ctl_mesh['verts'].shape[0], len(ctl_mesh['faces'])))
    print('victoria mesh: nverts = {0}, ntris = {1}'.format(tpl_mesh['verts'].shape[0], len(tpl_mesh['faces'])))

    out_path = f'{OUT_DIR}victoria_ctl.obj'
    export_mesh(out_path, ctl_mesh['verts'], ctl_mesh['faces'])

    out_path = f'{OUT_DIR}victoria_tpl.obj'
    export_mesh(out_path, tpl_mesh['verts'], tpl_mesh['faces'])


    for mdata_path in Path(M_DIR).glob('*.npy'):
        print(mdata_path)

        # load 2d measurements
        mdata = np.load(mdata_path )
        seg_dst_f = mdata.item().get('landmark_segment_dst_f')
        seg_dst_s = mdata.item().get('landmark_segment_dst_s')
        seg_location = mdata.item().get('landmark_segment_location_s')
        measurements = mdata.item().get('measurement')
        height = measurements['Height']

        arm_2d_f = mdata.item().get('armature_f')
        h_ratio = tpl_height / height
        arm_3d = scale_tpl_armature(arm_3d_tpl, arm_2d_f, h_ratio)

        id_mappings = slice_id_3d_2d_mappings()
        ct_mesh_slices = {}

        ctl_new_mesh = deepcopy(ctl_mesh)

        #hack: the background z value extracted from image is not exact. therefore, we consider ankle z as the z starting point
        tpl_ankle_hor = slc_id_locs['LAnkle'][1]
        tpl_ankle_ver = slc_id_locs['LAnkle'][2]
        #slice location in relative to ankle location in side image
        for id_3d, id_2d in id_mappings.items():
            #debug
            #if id_3d not in ['L0_RAnkle', 'L0_LAnkle']:
            #    continue

            #ignore the right body part
            if id_3d[0] == 'R':
                continue

            if id_3d == 'Shoulder':
                debug = True

            if id_3d  not in slc_id_vert_idxs:
                print(f'indices of {id_3d} are not available', file=sys.stderr)
                continue

            slc_idxs = slc_id_vert_idxs[id_3d]
            slice = ctl_mesh['verts'][slc_idxs]

            if id_2d not in seg_dst_f:
                print(f'measurement of {id_2d} is not available', file=sys.stderr)
                continue

            if id_2d not in seg_location:
                print(f'location of {id_2d} is not available. ignore this slice', file=sys.stderr)
                continue

            seg_log = seg_location[id_2d]

            w = seg_dst_f[id_2d]

            d = w
            if id_2d in seg_dst_s:
                d = seg_dst_s[id_2d]

            slc_loc_hor = seg_log[0]
            slc_loc_ver = np.abs(seg_log[1])

            #transform to victoria's scale
            w = w*h_ratio
            d = d*h_ratio
            slc_loc_ver = slc_loc_ver * h_ratio
            slc_loc_hor = slc_loc_hor * h_ratio

            #slc_org = slc_id_locs[id_3d]

            slice_out = copy(slice)
            #TEST
            if id_2d in models:
                if id_2d == 'Shoulder':
                    debug = True
                print('\t applied ', id_2d)
                model = models[id_2d]
                ratio = w/d
                pred = model.predict(np.reshape(ratio, (1,1)))[0, :]
                fourier = True
                if fourier:
                    res_contour = util.reconstruct_contour_fourier(pred)
                else:
                    if util.is_leg_contour(id_2d):
                        res_contour = util.reconstruct_leg_slice_contour(pred, d, w)
                    else:
                        res_contour = util.reconstruct_torso_slice_contour(pred, d, w, mirror=True)

                slice_out[:,0] =  res_contour[1,:]
                slice_out[:,1] =  res_contour[0,:]

                #right side
                if id_3d[0] == 'R':
                    #mirror through X
                    slice_out[:,0] = -slice_out[:,0]

                if util.is_leg_contour(id_2d):
                    slice_out += np.mean(slice, axis=0)

            dim_range = np.max(slice_out, axis=0) - np.min(slice_out, axis=0)
            w_ratio = w / dim_range[0]
            d_ratio = d / dim_range[1]
            slice_out = scale_vertical_slice(slice_out, w_ratio, d_ratio)

            if util.is_torso_contour(id_2d):
                if id_2d == 'Hip' or 'Crotch' in id_2d:
                    slice_out = fix_crotch_hip_cleavage(slice_out, d)

            if id_2d == 'UnderCrotch':
                plt.clf()
                plt.axes().set_aspect(1.0)
                plt.plot(res_contour[1, :], res_contour[0, :], '-r')
                plt.plot(res_contour[1, 0], res_contour[0, 0], '+r', ms=20)
                plt.plot(res_contour[1, 2], res_contour[0, 2], '+g', ms=20)

                plt.plot(slice_out[:, 0], slice_out[:, 1], '-b')
                plt.plot(slice_out[0, 0], slice_out[0, 1], '+r', ms=20)
                plt.plot(slice_out[2, 0], slice_out[2, 1], '+g', ms=20)

                plt.plot(slice[:, 0], slice[:, 1], '-y')
                plt.plot(slice[0, 0], slice[0, 1], '+r', ms=20)
                plt.plot(slice[2, 0], slice[2, 1], '+g', ms=20)
                #plt.savefig(f'{OUTPUT_DEBUG_DIR_TEST}{idx}.png')
                #plt.show()

            #align slice in vertical direction
            slice_out[:, 2] = slc_loc_ver + tpl_ankle_ver

            #align slice in horizontal direction
            #Warning: need to be careful here. we assume that the maximum point on hor dir is on the back side of Victoria's mesh
            slice_hor_anchor = np.max(slice_out[:,1])
            slice_out[:, 1] += (-slice_hor_anchor + tpl_ankle_hor + slc_loc_hor)

            ctl_new_mesh['verts'][slc_idxs, :] = slice_out

        #transform_arm_slices(ctl_new_mesh, slc_id_locs, slc_id_vert_idxs, arm_3d)

        verts = ctl_new_mesh['verts']
        for pair in ctl_sym_vert_pairs:
            mirror_co = deepcopy(verts[pair[0]])
            mirror_co[0] = -mirror_co[0]
            verts[pair[1]] = mirror_co

        ctl_mesh_quad_dom_new = deepcopy(ctl_mesh_quad_dom)
        ctl_mesh_quad_dom_new['verts'] = deepcopy(ctl_new_mesh['verts'])

        out_path = f'{OUT_DIR}{mdata_path.stem}_ctl.obj'
        print(f'\toutput control mesh: {out_path}')
        export_mesh(out_path, ctl_mesh_quad_dom_new['verts'], ctl_mesh_quad_dom_new['faces'])

        ctl_new_mesh_1 = subdivide_catmull_temp(ctl_mesh_quad_dom_new)
        out_path = f'{OUT_DIR}{mdata_path.stem}_ctl_subdivided.obj'
        print(f'\toutput subdivided control mesh: {out_path}')
        export_mesh(out_path, ctl_new_mesh_1['verts'], ctl_new_mesh_1['faces'])

        if Path(weight_path).is_file():
            with open(weight_path, 'rb') as f:
                data = pickle.load(f)
                ctl_tri_bs = data['control_mesh_tri_basis']
                vert_UVWs = data['template_vert_UVW']
                vert_weights = data['template_vert_weight']
                vert_effect_idxs = data['template_vert_effect_idxs']

            ctl_df_basis = util.calc_triangle_local_basis(ctl_new_mesh['verts'], ctl_new_mesh['faces'])
            tpl_df_mesh = deform_template_mesh(tpl_mesh, vert_effect_idxs, vert_weights, vert_UVWs, ctl_df_basis, deform_vert_idxs=None)

            out_path = f'{OUT_DIR}{mdata_path.stem}_deform.obj'
            print(f'\toutput deformed mesh to {out_path}')
            export_mesh(out_path, tpl_df_mesh['verts'], tpl_df_mesh['faces'])

            projec_mesh_onto_mesh(tpl_df_mesh, tpl_lbody_vert_idxs, ctl_new_mesh_1)

            #mirror
            verts = tpl_df_mesh['verts']
            for pair in tpl_sym_vert_pairs:
                mirror_co = deepcopy(verts[pair[0]])
                mirror_co[0] = -mirror_co[0]
                verts[pair[1]] = mirror_co

            out_path = f'{OUT_DIR}{mdata_path.stem}_deform_projected.obj'
            print(f'\toutput deformed mesh to {out_path}')
            export_mesh(out_path, tpl_df_mesh['verts'], tpl_df_mesh['faces'])


