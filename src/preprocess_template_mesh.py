import numpy as np
import argparse
import os
from pathlib import Path
import pickle
from src.obj_util import export_mesh
from src.util import  normalize
import src.util as util
from shapely.geometry import Polygon, Point

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

def map_slc_location_id(slc_rects, slc_locs):
    dst_tol = 0.01
    slc_locs_map = {}
    for i in range(slc_locs.shape[0]):
        loc = slc_locs[i, :]
        min_dst = 99999
        for id, rect in slc_rects.items():
            convex = Polygon(rect[:, :2]).convex_hull
            if convex.contains(Point(loc[:2])):
                pnt, norm = rect_plane(rect)
                dst = np.abs(dst_point_plane(loc, pnt, norm))
                if dst < min_dst:
                    min_dst = dst
                    min_id = id

        assert min_dst < dst_tol
        assert min_id not in slc_locs_map

        slc_locs_map[min_id] = loc

    return slc_locs_map

def extract_slice_verts_from_ctr_mesh(slc_rect, verts):
    dst_tolerance = 0.01

    pnt, norm = rect_plane(slc_rect)
    idxs = []
    for i in range(verts.shape[0]):
        co = verts[i, :]
        dst = np.abs(dst_point_plane(co, pnt, norm))
        if dst < dst_tolerance:
            idxs.append(i)

    slc_idxs = []
    convex = Polygon(slc_rect[:,:2]).convex_hull
    for idx in idxs:
        if convex.contains(Point(verts[idx,:2])):
            slc_idxs.append(idx)
    assert len(slc_idxs) == 7 or len(slc_idxs) == 8 or len(slc_idxs) == 14 or len(slc_idxs) == 20
    return np.array(slc_idxs)

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

def dst_point_plane(point, plane_point, plane_norm):
    return np.dot(plane_norm, point - plane_point)

def triangulate_quad_dominant_mesh(mesh):
    faces = mesh['faces']
    tris = []
    for i in range(len(faces)):
        face = faces[i]
        if len(face) == 4:
            tris.append((face[0], face[1], face[2]))
            tris.append((face[0], face[2], face[3]))
        elif len(face) == 3:
            tris.append(face)
    mesh['faces'] = tris
    return mesh

import concurrent.futures
def parameterize(verts, vert_effect_idxs, basis):
    print('parameterize')
    P = []
    T = np.zeros((3,3),np.float32)
    for i in range(verts.shape[0]):
        v_co = verts[i,:]
        idxs = vert_effect_idxs[i]
        v_uvw = []
        for ctl_tri_idx in idxs:
            d = v_co - basis[ctl_tri_idx, 0, :]
            T[:,0] = basis[ctl_tri_idx, 1, :]
            T[:,1] = basis[ctl_tri_idx, 2, :]
            T[:,2] = basis[ctl_tri_idx, 3, :]
            local_co = np.linalg.solve(T, d)
            v_uvw.append(local_co)
        P.append(v_uvw)
        if i % 500 == 0:
            print(i,'/',verts.shape[0])

    return P


def vert_tri_weight(v, t_v0, t_v1, t_v2):
    beta = 1.5
    center = (t_v0+t_v1+t_v2)/3.0
    d = np.linalg.norm(v-center)
    l = (np.linalg.norm(t_v0-center) + np.linalg.norm(t_v1-center) + np.linalg.norm(t_v2-center))/3.0
    ratio = d/l
    if ratio < beta:
        return 1.0-ratio/beta
    else:
        return 0.0

def calc_vertex_weigth_control_mesh(verts, verts_ref, tris_ref):
    print('calc_vertex_weigth_control_mesh')
    effect_idxs = []
    effect_weights = []

    tri_centers = np.zeros((len(tris_ref), 3), dtype=np.float32)
    tri_radius = np.zeros((len(tris_ref)), dtype=np.float32)
    for j, t in enumerate(tris_ref):
        v0, v1, v2 = verts_ref[t[0]], verts_ref[t[1]], verts_ref[t[2]]
        center = (v0+v1+v2)/3.0
        tri_centers[j, :] = center
        tri_radius[j] = (np.linalg.norm(v0-center) + np.linalg.norm(v1-center) + np.linalg.norm(v2-center))/3.0

    beta = 2
    for i in range(verts.shape[0]):
        v = verts[i,:]
        idxs = []
        weights = []

        #debug
        d_min = 99999999
        w_min = 0
        d_min_idx = 0
        for j,t in enumerate(tris_ref):
            #w = vert_tri_weight(v, verts_ref[t[0]], verts_ref[t[1]], verts_ref[t[2]])
            d = np.linalg.norm(v-tri_centers[j,:])
            ratio = d/tri_radius[j]
            if ratio < beta:
                w = 1.0 - ratio/beta
                idxs.append(j)
                weights.append(w)
                if d < d_min:
                    d_min = d
                    w_min = w
                    d_min_idx = j

        if i % 500 == 0:
            print(i,'/',verts.shape[0])
        effect_idxs.append(idxs)
        effect_weights.append(weights)
        #effect_idxs.append([d_min_idx])
        #effect_weights.append([w_min])

    return effect_idxs, effect_weights

def rect_plane(rect):
    n = np.cross(rect[2]-rect[0], rect[1]-rect[0])
    n = n / np.linalg.norm(n)
    p = np.mean(rect, axis=0)
    return p,n

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vic", "--victoria", required=True, help="victoria pkl file")
    ap.add_argument("-o", "--out_path", required=True, help="directory for expxorting control mesh slices")
    args = vars(ap.parse_args())

    vic_path = args['victoria']
    out_path = args['out_path']

    slice_locs = None
    vic_height = None
    with open(vic_path, 'rb') as f:
        data = pickle.load(f)
        slice_locs = data['slice_locs']
        vic_seg = data['height_segment']
        vic_height = np.linalg.norm(vic_seg)
        slc_id_rects = data['slice_rects']
        ctl_mesh = data['ctl_mesh']
        tpl_mesh = data['vic_mesh']

    ctl_mesh = triangulate_quad_dominant_mesh(ctl_mesh)
    tpl_mesh = triangulate_quad_dominant_mesh(tpl_mesh)

    print('control  mesh: nverts = {0}, ntris = {1}'.format(ctl_mesh['verts'].shape[0], len(ctl_mesh['faces'])))
    print('victoria mesh: nverts = {0}, ntris = {1}'.format(tpl_mesh['verts'].shape[0], len(tpl_mesh['faces'])))

    ctl_tri_bs = util.calc_triangle_local_basis(ctl_mesh['verts'], ctl_mesh['faces'])
    vert_effect_idxs, vert_weights = calc_vertex_weigth_control_mesh(tpl_mesh['verts'], ctl_mesh['verts'], ctl_mesh['faces'])
    #vert_effect_idxs, vert_weights = calc_vertex_weigth_control_mesh_parallel(tpl_mesh['verts'], ctl_mesh['verts'], ctl_mesh['faces'])
    vert_UVW = parameterize(tpl_mesh['verts'], vert_effect_idxs, ctl_tri_bs)

    slc_id_locs = map_slc_location_id(slc_id_rects, slice_locs)

    slc_id_vert_idxs= {}
    for id_3d, rect in slc_id_rects.items():
        idxs = extract_slice_verts_from_ctr_mesh(rect, ctl_mesh['verts'])
        assert id_3d not in slc_id_vert_idxs
        slc_id_vert_idxs[id_3d] = idxs

    out_data = {}
    out_data['control_mesh'] = ctl_mesh
    out_data['control_mesh_tri_basis'] = ctl_tri_bs

    out_data['template_mesh'] = tpl_mesh
    out_data['template_vert_UVW'] = vert_UVW
    out_data['template_vert_weight'] = vert_weights
    out_data['template_vert_effect_idxs'] = vert_effect_idxs

    out_data['template_height']  = vic_height
    out_data['slice_locs']  = slc_id_locs
    out_data['slice_vert_idxs']  = slc_id_vert_idxs

    with open(out_path, 'wb') as f:
        pickle.dump(out_data, f)
