from scipy.interpolate import splprep, splev
from scipy.fftpack import fft, ifft
import numpy as  np
import bpy
import os
import pickle
import mathutils.geometry as geo
from mathutils import Vector
from bpy import context
from collections import defaultdict
from copy import deepcopy

def mesh_to_numpy(mesh):
    nverts = len(mesh.vertices)
    verts = []
    for iv in range(nverts):
        verts.append(mesh.vertices[iv].co[:])
    
    faces = []
    for p in mesh.polygons:
        faces.append(p.vertices[:])

    edges = []
    for e in mesh.edges:
        edges.append(e.vertices[:])

    v_e = [[] for _ in range(nverts)]
    nedges = len(mesh.edges)
    for ie in range(nedges):
        e = mesh.edges[ie]
        iv_0 = e.vertices[0]
        iv_1 = e.vertices[1]
        v_e[iv_0].append(ie)
        v_e[iv_1].append(ie)

    v_f = [[] for _ in range(nverts)]
    nfaces = len(mesh.polygons)
    for ip in range(nfaces):
        p = mesh.polygons[ip]
        for iv in p.vertices:
            v_f[iv].append(ip)

    e_f = [[] for _ in range(nedges)]
    for ip in range(nfaces):
        p = mesh.polygons[ip]
        for il in p.loop_indices:
            l = mesh.loops[il]
            e_f[l.edge_index].append(ip)

    loops = [(l.vertex_index, l.edge_index) for l in mesh.loops]
    f_l = [[] for _ in range(nfaces)]
    for ip in range(nfaces):
        p = mesh.polygons[ip]
        for il in p.loop_indices:
            f_l[ip].append(il)

    mesh = {'verts':np.array(verts), 'faces':faces, 'edges': edges, 'loops': loops, 'f_l':f_l, 'v_e':v_e, 'v_f':v_f, 'e_f': e_f}

    return mesh

def clockwiseangle_and_distance(point, org):
    refvec = np.array([0, 1])
    # Vector between point and the origin: v = p - o
    vector = [point[0] - org[0], point[1] - org[1]]
    # Length of vector: ||v||
    lenvector = np.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -np.pi, 0

    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]

    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = np.arctan2(diffprod, dotprod)

    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * np.pi + angle, lenvector

    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


def arg_sort_points_cw(points):
    center = (np.mean(points[:, 0]), np.mean(points[:, 1]))
    compare_func = lambda pair: clockwiseangle_and_distance(pair[1], center)
    points = sorted(enumerate(points), key=compare_func)
    return [pair[0] for pair in points[::-1]]


def sort_leg_slice_vertices(slc_vert_idxs, mesh_verts):
    X = mesh_verts[slc_vert_idxs][:, 1]
    Y = mesh_verts[slc_vert_idxs][:, 0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    points_0 = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    # find the starting point of the leg contour
    # the contour must start at that point to match the order of the prediction contour
    # check the leg contour in the blender file for why it is this way
    start_idx = np.argmin(points_0[:, 1]) + 2
    points_0 = np.roll(points_0, axis=0, shift=-start_idx)

    # concatenate two sorted part.
    sorted_points = points_0

    # map indices
    slc_sorted_vert_idxs = []
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i, :]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        assert closest_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(slc_vert_idxs[closest_idx])

    return slc_sorted_vert_idxs


def sort_torso_slice_vertices(slc_vert_idxs, mesh_verts, title=''):
    X = mesh_verts[slc_vert_idxs][:, 1]
    Y = mesh_verts[slc_vert_idxs][:, 0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    # we needs to split our point array into two part because the clockwise sort just works on convex polygon. it will fail at strong concave points at crotch slice
    # sort the upper part
    mask_0 = Y >= -0.01
    X_0 = X[mask_0]
    Y_0 = Y[mask_0]
    assert (len(X_0) > 0 and len(Y_0) > 0)
    points_0 = np.concatenate([X_0[:, np.newaxis], Y_0[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    #print("\t\tpart one: ", arg_points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    # find the first point of the contour
    # the contour must start at that point to match the order of the prediction contour
    # check the leg contour in the blender file for why it is this way
    min_y = np.inf
    min_y_idx = 0
    for i in range(points_0.shape[0]):
        if points_0[i, 0] > 0:
            if points_0[i, 1] < min_y:
                min_y = points_0[i, 1]
                min_y_idx = i
    points_0 = np.roll(points_0, axis=0, shift=-min_y_idx)

    # sort the below part
    mask_1 = ~mask_0
    X_1 = X[mask_1]
    Y_1 = Y[mask_1]
    assert (len(X_1) > 0 and len(Y_1) > 0)
    points_1 = np.concatenate([X_1[:, np.newaxis], Y_1[:, np.newaxis]], axis=1)
    arg_points_1 = arg_sort_points_cw(points_1)
    #print("\t\tpart two: ", arg_points_1)
    points_1 = np.array(points_1[arg_points_1, :])

    # concatenate two sorted part.
    sorted_points = np.concatenate([points_0, points_1], axis=0)
    # map indices
    slc_sorted_vert_idxs = []
    #print("mapping points")
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i, :]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        found_idx = slc_vert_idxs[closest_idx]
        assert found_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(found_idx)
        
    return slc_sorted_vert_idxs


def is_torso_slice(id):
    torso_slc_ids = {'Crotch', 'Aux_Crotch_Hip_0', 'Aux_Crotch_Hip_1', 'Aux_Crotch_Hip_2', 'Hip',
                     'Aux_Hip_Waist_0', 'Aux_Hip_Waist_1', 'Waist',
                     'Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1', 'Aux_Waist_UnderBust_1', 'UnderBust',
                     'Aux_UnderBust_Bust_0', 'Bust', 'Armscye', 'Aux_Armscye_Shoulder_0', 'Shoulder'}
    if id in torso_slc_ids:
        return True
    else:
        return False


def is_leg_slice(id):
    leg_slc_ids = {'LKnee', 'RKnee', 'LUnderCrotch', 'RUnderCrotch', 'LAux_Knee_UnderCrotch_3',
                   'LAux_Knee_UnderCrotch_2', 'LAux_Knee_UnderCrotch_1', 'LAux_Knee_UnderCrotch_0'}
    if id in leg_slc_ids:
        return True
    else:
        return False

def resample_contour(X, Y, n_point):
    tck, u = splprep([X, Y], s=0)
    u_1 = np.linspace(0.0, 1.0, n_point)
    X, Y = splev(u_1, tck)
    return X, Y

def adjust_slice_vertices_density(slc_idxs, mesh_verts):
    n_points = len(slc_idxs)
    slc_verts = mesh_verts[slc_idxs, :]
    X = slc_verts[:,0]
    Y = slc_verts[:,1]
    X = np.array(list(X)+[X[0]])
    Y = np.array(list(Y)+[Y[0]])
    X, Y = resample_contour(X, Y, 200)

    cnt_complex = np.array([np.complex(x,y) for x, y in zip(X,Y)])
    tf_0 = fft(cnt_complex)

    half = int(n_points/2)
    if n_points % 2 == 0:
        tf_1 = np.concatenate([tf_0[0:half], tf_0[-half:]])
    else:
        tf_1 = np.concatenate([tf_0[0:half+1], tf_0[-half:]])

    res_contour = ifft(tf_1)
    res_contour = np.concatenate([np.real(res_contour).reshape(-1, 1), np.imag(res_contour).reshape(-1, 1)], axis=1)

    res_range_x = np.max(res_contour[:, 0]) - np.min(res_contour[:, 0])
    range_x = np.max(X) - np.min(X)
    scale_x = range_x / res_range_x

    res_range_y = np.max(res_contour[:, 1]) - np.min(res_contour[:, 1])
    range_y = np.max(Y) - np.min(Y)
    scale_y = range_y / res_range_y

    res_contour *= max(scale_x, scale_y)
    
    for i in range(n_points):
        slc_verts[i,:][:2] = res_contour[i,:]
    
    z_cos = slc_verts[:,2]
    z = np.mean(z_cos)
    slc_verts[:,2] = z
    
    return slc_verts
        
def extract_slice_vert_indices(ctl_obj):
    print(ctl_obj.name)
    mesh = ctl_obj.data

    mdata = mesh_to_numpy(mesh)
    ctl_verts = mdata['verts']

    slc_vert_idxs = defaultdict(list)
    for v in mesh.vertices:
        # assert len(v.groups) == 1
        for vgrp in v.groups:
            grp_name = ctl_obj.vertex_groups[vgrp.group].name
            slc_vert_idxs[grp_name].append(v.index)
    output = {}
    for slc_id, idxs in slc_vert_idxs.items():
        output[slc_id] = np.array(idxs)
    
    print('sortinng slice vertices counter clockwise, starting from the extreme point on the +X axis')
    for id, slc_idxs in output.items():
        slc_idxs_1 = None
        if is_leg_slice(id):
            print('\t\t sort slice: ', id)
            slc_idxs_1 = sort_leg_slice_vertices(slc_idxs, ctl_verts)
            output[id] = slc_idxs
        elif is_torso_slice(id):
            print('\t\t sort slice: ', id)
            slc_idxs_1 = sort_torso_slice_vertices(slc_idxs, ctl_verts, title=id)
            output[id] = slc_idxs
        else:
            print('\t\t ignore slice: ', id)
        
        if slc_idxs_1 is not None:
            slc_new_co = adjust_slice_vertices_density(slc_idxs_1, ctl_verts)
            for i, idx in enumerate(slc_idxs_1):
                co = slc_new_co[i,:]
                mesh.vertices[idx].co[:] = co[:]
    
    mid_vert_idxs = slc_vert_idxs['MidVertices']
    for idx in mid_vert_idxs:
        mesh.vertices[idx].co.x = 0.0
        
    return output

#bpy.ops.object.vertex_group_remove_from(use_all_groups=True)
print("hello khanh")
ctl_obj = bpy.data.objects["ControlMesh"]
extract_slice_vert_indices(ctl_obj)