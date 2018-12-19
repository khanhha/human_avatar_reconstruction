import numpy as np
import argparse
from collections import defaultdict
import os
from pathlib import Path
import pickle
from copy import deepcopy
from src.util import  normalize
import matplotlib.pyplot as plt
import src.util as util
from shapely.geometry import Polygon, Point
import multiprocessing
from functools import partial
from scipy import stats

from enum import Enum
class BDPart(Enum):
    Part_Head = 1
    Part_LArm = 2
    Part_RArm = 3
    Part_LLeg = 4
    Part_RLeg = 5
    Part_Torso = 6

def dst_point_plane(point, plane_point, plane_norm):
    return np.dot(plane_norm, point - plane_point)

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

def average_circumscribed_radius_tris(verts, tris, tris_idxs):
    rads = np.zeros(len(tris_idxs), dtype=np.float32)
    for j, t_idx in enumerate(tris_idxs):
        v0, v1, v2 = verts[tris[t_idx][0]], verts[tris[t_idx][1]], verts[tris[t_idx][2]]
        center = (v0+v1+v2)/3.0
        rads[j] = (np.linalg.norm(v0-center) + np.linalg.norm(v1-center) + np.linalg.norm(v2-center))/3.0
    return np.mean(rads)

def is_adjacent_tri(t0, t1):
    for i in range(3):
        v00, v01 = t0[i], t0[(i+1)%3]
        for j in range(3):
            v10, v11 = t1[j], t1[(j+1)%3]
            if (v00 == v10 and v01 == v11) or (v00 == v11 and v01 == v10):
                return True
    return False

def find_neighbour_body_part_triangles(tris_ctl, ctl_tri_bd_parts):
    n_tris = len(tris_ctl)
    bd_part_adj_tri_idxs = defaultdict(list)

    for tri_idx, part_id in enumerate(ctl_tri_bd_parts):
        for other_tri_idx in range(n_tris):
            #not the triangle we are processing
            if tri_idx == other_tri_idx:
                continue

            #not in the same body part
            if ctl_tri_bd_parts[other_tri_idx] == part_id:
                continue

            if is_adjacent_tri(tris_ctl[tri_idx], tris_ctl[other_tri_idx]):
                bd_part_adj_tri_idxs[part_id].append(other_tri_idx)

    return bd_part_adj_tri_idxs

def calc_vertex_weigth_control_mesh_local(verts_tpl, verts_ctl, tris_ctl, tpl_v_body_parts, ctl_tri_bd_parts, effective_range_factor = 6):

    effect_idxs = [None]* len(verts_tpl)
    effect_weights =[None]* len(verts_tpl)

    ctl_tri_centers = np.zeros((len(tris_ctl), 3), dtype=np.float32)
    for j, t in enumerate(tris_ctl):
        v0, v1, v2 = verts_ctl[t[0]], verts_ctl[t[1]], verts_ctl[t[2]]
        center = (v0+v1+v2)/3.0
        ctl_tri_centers[j, :] = center

    bd_part_ctl_tris_idxs = defaultdict(list)
    for tri_idx, _ in enumerate(tris_ctl):
        part_id = ctl_tri_bd_parts[tri_idx]
        bd_part_ctl_tris_idxs[part_id].append(tri_idx)

    print('generating neighbouring information for control mesh')
    bd_part_adj_ctl_tri_idxs = find_neighbour_body_part_triangles(tris_ctl, ctl_tri_bd_parts)

    bd_part_ctl_avg_rads = {id : average_circumscribed_radius_tris(verts_ctl, tris_ctl, part_tris_idxs) for id, part_tris_idxs in bd_part_ctl_tris_idxs.items()}
    print(bd_part_ctl_avg_rads )

    body_part_tpl_verts = {}
    for i, part_id in enumerate(tpl_v_body_parts):
        if part_id not in body_part_tpl_verts:
            body_part_tpl_verts[part_id] = []
        body_part_tpl_verts[part_id].append(i)

    print('start calculating weights')
    cnt_progress = 0
    n_zero_weights = 0
    for part_id, verts  in body_part_tpl_verts.items():
        if part_id not in bd_part_ctl_avg_rads:
            print(f'missing body part of ID {part_id}')
            continue
        print(f'processing body part {part_id}')
        bd_part_avg_rad = bd_part_ctl_avg_rads[part_id]
        for v_i in  verts:
            tri_idxs = []
            weights = []
            v_co = verts_tpl[v_i, :]

            #weights respect to control tringle in the same body part
            ctl_tris_idxs = bd_part_ctl_tris_idxs[part_id]
            for tri_idx in ctl_tris_idxs:
                d = np.linalg.norm(v_co - ctl_tri_centers[tri_idx, :])
                ratio = d / bd_part_avg_rad
                if ratio < effective_range_factor:
                    w = ratio / effective_range_factor
                    w = 1 - w
                    #w = np.exp(-4.0*w*w)
                    tri_idxs.append(tri_idx)
                    weights.append(w)

            #weights respect to neighbour control triangle in the adjacent body part
            ctl_tris_idxs = bd_part_adj_ctl_tri_idxs[part_id]
            for tri_idx in ctl_tris_idxs:
                d = np.linalg.norm(v_co - ctl_tri_centers[tri_idx, :])
                ratio = d / bd_part_avg_rad
                if ratio < effective_range_factor:
                    w = 1.0 - ratio / effective_range_factor
                    tri_idxs.append(tri_idx)
                    weights.append(w)

            cnt_progress += 1
            if len(tri_idxs) == 0:
                n_zero_weights += 1
            if cnt_progress % 500 == 0:
                print(cnt_progress, '/', verts_tpl.shape[0], 'zero weight vertices = ', n_zero_weights)

            effect_idxs[v_i] = tri_idxs
            effect_weights[v_i] = weights

    return effect_idxs, effect_weights

def vertex_process_global(v, tris_ref, tri_centers, tri_radius, effective_range_factor):
    idxs = []
    weights = []
    for j, t in enumerate(tris_ref):
        #w = vert_tri_weight(v, verts_ref[t[0]], verts_ref[t[1]], verts_ref[t[2]])
        d = np.linalg.norm(v - tri_centers[j, :])
        ratio = d/tri_radius[j]
        if ratio < effective_range_factor:
            #w = 1.0 - ratio / effective_range_factor
            w = ratio/effective_range_factor
            w = 1.0 - w
            #w = np.exp(w*w)
            idxs.append(j)
            weights.append(w)

    return (idxs, weights)

def calc_vertex_weigth_control_mesh_global(verts, verts_ref, tris_ref, effective_range_factor = 4):
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

    nprocess = 12
    pool = multiprocessing.Pool(nprocess)
    results = pool.map(func=partial(vertex_process_global, tris_ref=tris_ref, tri_centers=tri_centers, tri_radius =tri_radius, effective_range_factor=effective_range_factor),
                       iterable=verts, chunksize=128)

    # for i in range(verts.shape[0]):
    #     v = verts[i,:]
    #     idxs = []
    #     weights = []
    #
    #     for j,t in enumerate(tris_ref):
    #         #w = vert_tri_weight(v, verts_ref[t[0]], verts_ref[t[1]], verts_ref[t[2]])
    #         d = np.linalg.norm(v-tri_centers[j,:])
    #         ratio = d/tri_radius[j]
    #         if ratio < effective_range_factor:
    #             #w = 1.0 - ratio / effective_range_factor
    #             w = ratio/effective_range_factor
    #             w = np.exp(-6.0*w*w)
    #             idxs.append(j)
    #             weights.append(w)
    #
    #     if i % 500 == 0:
    #         print(i,'/',verts.shape[0])
    #     effect_idxs.append(idxs)
    #     effect_weights.append(weights)
    #     effect_idxs.append([d_min_idx])
    #     effect_weights.append([w_min])
    for pair in results:
        effect_idxs.append(pair[0])
        effect_weights.append(pair[1])

    lens = np.array([len(idxs) for idxs in effect_idxs])
    stat = stats.describe(lens)
    print(f'finish calculating weights')
    print(f'\tneighbor size statistics:')
    print(f'\t{stat}')

    return effect_idxs, effect_weights

def rect_plane(rect):
    n = np.cross(rect[2]-rect[0], rect[1]-rect[0])
    n = n / np.linalg.norm(n)
    p = np.mean(rect, axis=0)
    return p,n

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vic", "--victoria", required=True, help="victoria pkl file")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    ap.add_argument("-g", "--global", required=True, help="global weight")
    args = vars(ap.parse_args())

    vic_path = args['victoria']
    out_path = args['out_dir']
    is_global = int(args['global']) > 0

    with open(vic_path, 'rb') as f:
        data = pickle.load(f)
        slc_id_locs = data['slice_locs']
        arm_bone_locs = data['arm_bone_locs']
        slice_id_vert_idxs = data['slice_vert_idxs']

        ctl_mesh = data['control_mesh']
        ctl_mesh_quad_dom = data['control_mesh_quad_dom']

        tpl_mesh = data['template_mesh']

    print('control  mesh: nverts = {0}, nfaces = {1}'.format(ctl_mesh['verts'].shape[0], len(ctl_mesh['faces'])))
    print('victoria mesh: nverts = {0}, nfaces = {1}'.format(tpl_mesh['verts'].shape[0], len(tpl_mesh['faces'])))

    print('calculating weight')
    ctl_tri_bs = util.calc_triangle_local_basis(ctl_mesh['verts'], ctl_mesh['faces'])

    if is_global:
        vert_effect_idxs, vert_weights = calc_vertex_weigth_control_mesh_global(tpl_mesh['verts'], ctl_mesh['verts'], ctl_mesh['faces'], effective_range_factor=2)
    else:
        ctl_f_body_parts = data['control_mesh_face_body_parts']
        tpl_v_body_parts = data['template_vert_body_parts']
        body_part_dict = data['body_part_dict']
        vert_effect_idxs, vert_weights = calc_vertex_weigth_control_mesh_local(tpl_mesh['verts'], ctl_mesh['verts'], ctl_mesh['faces'], tpl_v_body_parts, ctl_f_body_parts)

    vert_UVW = parameterize(tpl_mesh['verts'], vert_effect_idxs, ctl_tri_bs)

    w_data = {}
    w_data['template_vert_UVW'] = vert_UVW
    w_data['template_vert_weight'] = vert_weights
    w_data['template_vert_effect_idxs'] = vert_effect_idxs
    w_data['control_mesh_tri_basis'] = ctl_tri_bs

    with open(out_path, 'wb') as f:
        pickle.dump(w_data, f)

