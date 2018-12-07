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

from enum import Enum
class BDPart(Enum):
    Part_Head = 1
    Part_LArm = 2
    Part_RArm = 3
    Part_LLeg = 4
    Part_RLeg = 5
    Part_Torso = 6

def clockwiseangle_and_distance(point, org):
    refvec = np.array([0, 1])
    # Vector between point and the origin: v = p - o
    vector = [point[0]-org[0], point[1]-org[1]]
    # Length of vector: ||v||
    lenvector = np.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -np.pi, 0

    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]

    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = np.arctan2(diffprod, dotprod)

    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*np.pi+angle, lenvector

    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

def arg_sort_points_cw(points):
    center = (np.mean(points[:,0]), np.mean(points[:,1]))
    compare_func = lambda pair: clockwiseangle_and_distance(pair[1], center)
    points = sorted(enumerate(points), key = compare_func)
    return [pair[0] for pair in points[::-1]]

def sort_leg_slice_vertices(slc_vert_idxs, mesh_verts):
    X =  mesh_verts[slc_vert_idxs][:,1]
    Y =  mesh_verts[slc_vert_idxs][:,0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    points_0 = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    #find the starting point of the leg contour
    #the contour must start at that point to match the order of the prediction contour
    #check the leg contour in the blender file for why it is this way
    start_idx  = np.argmin(points_0[:,1])+1
    points_0 = np.roll(points_0, axis=0, shift=-start_idx)

    #concatenate two sorted part.
    sorted_points = points_0

    #map indices
    slc_sorted_vert_idxs = []
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i,:]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        assert closest_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(slc_vert_idxs[closest_idx])

    sorted_X =  mesh_verts[slc_sorted_vert_idxs][:,0]
    sorted_Y =  mesh_verts[slc_sorted_vert_idxs][:,1]
    #plt.clf()
    #plt.axes().set_aspect(1)
    #plt.plot(points_0[:,0], points_0[:,1], '+r')
    #plt.plot(sorted_points[:5,0], sorted_points[:5,1],'-b')
    #plt.plot(sorted_X, sorted_Y,'-r')
    #plt.show()
    return slc_sorted_vert_idxs

def sort_torso_slice_vertices(slc_vert_idxs, mesh_verts, title =''):
    X =  mesh_verts[slc_vert_idxs][:,1]
    Y =  mesh_verts[slc_vert_idxs][:,0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    #we needs to split our point array into two part because the clockwise sort just works on convex polygon. it will fail at strong concave points at crotch slice
    #sort the upper part
    mask_0 = Y >= -0.01
    X_0 = X[mask_0]
    Y_0 = Y[mask_0]
    assert (len(X_0) > 0 and len(Y_0) > 0)
    points_0 = np.concatenate([X_0[:, np.newaxis], Y_0[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    #find the first point of the contour
    #the contour must start at that point to match the order of the prediction contour
    #check the leg contour in the blender file for why it is this way
    min_y = np.inf
    min_y_idx = 0
    for i in range(points_0.shape[0]):
        if points_0[i,0] > 0:
            if points_0[i,1] < min_y:
                min_y = points_0[i,1]
                min_y_idx = i
    points_0 = np.roll(points_0, axis=0, shift=-min_y_idx)

    #sort the below part
    mask_1 = ~mask_0
    X_1 = X[mask_1]
    Y_1 = Y[mask_1]
    assert (len(X_1) > 0 and len(Y_1) > 0)
    points_1 = np.concatenate([X_1[:, np.newaxis], Y_1[:, np.newaxis]], axis=1)
    arg_points_1 = arg_sort_points_cw(points_1)
    points_1 = np.array(points_1[arg_points_1, :])

    #concatenate two sorted part.
    sorted_points = np.concatenate([points_0, points_1], axis=0)

    #map indices
    slc_sorted_vert_idxs = []
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i,:]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        assert closest_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(slc_vert_idxs[closest_idx])

    sorted_X =  mesh_verts[slc_sorted_vert_idxs][:,0]
    sorted_Y =  mesh_verts[slc_sorted_vert_idxs][:,1]
    # plt.clf()
    # plt.axes().set_aspect(1)
    # plt.plot(points_0[:,0], points_0[:,1], '+r')
    # plt.plot(sorted_points[:,0], sorted_points[:,1],'-b')
    # plt.plot(sorted_X, sorted_Y,'-r')
    # plt.title(title)
    #plt.show()
    return slc_sorted_vert_idxs

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

def triangulate_quad_dominant_mesh_1(mesh, face_flags):
    faces = mesh['faces']
    tris = []
    tri_flags = []
    for face, flag in zip(faces,face_flags):
        if len(face) == 4:
            tris.append((face[0], face[1], face[2]))
            tri_flags.append(flag)
            tris.append((face[0], face[2], face[3]))
            tri_flags.append(flag)
        elif len(face) == 3:
            tris.append(face)
            tri_flags.append(flag)
        else:
            assert 'unpredicted face length {0}'.format(len(face))

    mesh['faces'] = tris
    return mesh, tri_flags

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

def calc_vertex_weigth_control_mesh_local(verts_tpl, verts_ctl, tris_ctl, tpl_v_body_parts, ctl_tri_bd_parts, effective_range_factor = 2):

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

def calc_vertex_weigth_control_mesh_global(verts, verts_ref, tris_ref, effective_range_factor = 3.0):
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

    for i in range(verts.shape[0]):
        v = verts[i,:]
        idxs = []
        weights = []

        for j,t in enumerate(tris_ref):
            #w = vert_tri_weight(v, verts_ref[t[0]], verts_ref[t[1]], verts_ref[t[2]])
            d = np.linalg.norm(v-tri_centers[j,:])
            ratio = d/tri_radius[j]
            if ratio < effective_range_factor:
                #w = 1.0 - ratio / effective_range_factor
                w = ratio/effective_range_factor
                w = np.exp(-6.0*w*w)
                idxs.append(j)
                weights.append(w)

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
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    ap.add_argument("-w", "--weight_update", required=True, help="update weight or not: it takes several minute")
    args = vars(ap.parse_args())

    vic_path = args['victoria']
    OUT_DIR = args['out_dir']
    update_weight = int(args['weight_update'])

    slc_id_locs = None
    vic_height = None
    with open(vic_path, 'rb') as f:
        data = pickle.load(f)
        slc_id_locs = data['slice_locs']
        arm_bone_locs = data['arm_bone_locs']
        vic_seg = data['height_segment']
        vic_height = np.linalg.norm(vic_seg)
        slice_id_vert_idxs = data['slice_vert_idxs']

        ctl_mesh = data['ctl_mesh']
        ctl_mesh_quad_dom = data['ctl_mesh_quad_dom']
        ctl_f_body_parts = data['ctl_f_body_parts']

        tpl_mesh = data['vic_mesh']
        tpl_v_body_parts = data['vic_v_body_parts']

        body_part_dict = data['body_part_dict']

    print('control  mesh: nverts = {0}, nfaces = {1}'.format(ctl_mesh['verts'].shape[0], len(ctl_mesh['faces'])))
    print('victoria mesh: nverts = {0}, nfaces = {1}'.format(tpl_mesh['verts'].shape[0], len(tpl_mesh['faces'])))

    print('sortinng slice vertices counter clockwise, starting from the extreme point on the +X axis')
    sort_ids = ['Crotch', 'Aux_Crotch_Hip_0', 'Aux_Crotch_Hip_1', 'Hip', 'Waist', 'UnderBust', 'Aux_Hip_Waist_0', 'Aux_Hip_Waist_1',
                'Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1', 'Aux_Waist_UnderBust_2',  'Aux_UnderBust_Bust_0', 'Bust', 'Armscye']
    for id, slc_idxs in slice_id_vert_idxs.items():
        if id in sort_ids:
            print(f'\t\t{id}')
            slc_idxs = sort_torso_slice_vertices(slc_idxs, ctl_mesh['verts'], title=id)
            slice_id_vert_idxs[id] = slc_idxs

    sort_ids_1 = ['LKnee', 'RKnee', 'LUnderCrotch', 'RUnderCrotch', 'LAux_Knee_UnderCrotch_3', 'LAux_Knee_UnderCrotch_2', 'LAux_Knee_UnderCrotch_1', 'LAux_Knee_UnderCrotch_0']
    for id, slc_idxs in slice_id_vert_idxs.items():
        if id in sort_ids_1:
            print(f'\t\t{id}')
            slc_idxs = sort_leg_slice_vertices(slc_idxs, ctl_mesh['verts'])
            slice_id_vert_idxs[id] = slc_idxs

    n_quad = len(ctl_mesh['faces'])
    ctl_mesh_quad = deepcopy(ctl_mesh)
    ctl_mesh, ctl_f_body_parts  = triangulate_quad_dominant_mesh_1(ctl_mesh, ctl_f_body_parts)
    n_tris = len(ctl_mesh['faces'])
    print(f'triangulate {n_tris - n_quad} quads of the control mesh')

    tpl_mesh = triangulate_quad_dominant_mesh(tpl_mesh)

    print('control  mesh: nverts = {0}, ntris = {1}'.format(ctl_mesh['verts'].shape[0], len(ctl_mesh['faces'])))
    print('victoria mesh: nverts = {0}, ntris = {1}'.format(tpl_mesh['verts'].shape[0], len(tpl_mesh['faces'])))

    ctl_tri_bs = util.calc_triangle_local_basis(ctl_mesh['verts'], ctl_mesh['faces'])
    if update_weight > 0:
        print('calculating weight')
        #vert_effect_idxs, vert_weights = calc_vertex_weigth_control_mesh_global(tpl_mesh['verts'], ctl_mesh['verts'], ctl_mesh['faces'])
        vert_effect_idxs, vert_weights = calc_vertex_weigth_control_mesh_local(tpl_mesh['verts'], ctl_mesh['verts'], ctl_mesh['faces'], tpl_v_body_parts, ctl_f_body_parts)
        vert_UVW = parameterize(tpl_mesh['verts'], vert_effect_idxs, ctl_tri_bs)

        w_data = {}
        w_data['template_vert_UVW'] = vert_UVW
        w_data['template_vert_weight'] = vert_weights
        w_data['template_vert_effect_idxs'] = vert_effect_idxs

        with open(f'{OUT_DIR}/vic_weight.pkl', 'wb') as f:
            pickle.dump(w_data, f)
    else:
        print('weight calculation is ignored')

    out_data = {}
    out_data['control_mesh'] = ctl_mesh
    out_data['control_mesh_quad_dom'] = ctl_mesh_quad_dom
    out_data['control_mesh_tri_basis'] = ctl_tri_bs

    out_data['template_mesh'] = tpl_mesh


    out_data['template_height']  = vic_height
    out_data['slice_locs']  = slc_id_locs
    out_data['slice_vert_idxs']  = slice_id_vert_idxs

    out_data['arm_bone_locs']  = arm_bone_locs

    with open(f'{OUT_DIR}/vic_data.pkl', 'wb') as f:
        pickle.dump(out_data, f)
