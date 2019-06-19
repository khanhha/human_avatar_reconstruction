import cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt
from common.obj_util import import_mesh_tex_obj, export_mesh_tex_obj
import sys
sys.path.insert(0, '/home/khanhhh/data_1/sample_codes/libigl/python')
import pyigl as igl

def find_head_tex_verts(mesh, head_vert_idxs):
    head_vert_set = set(head_vert_idxs)
    head_tex_verts_set = set()
    ft = mesh['ft']
    faces = mesh['f']
    for f_idx, f in enumerate(faces):
        is_head = True
        for v_idx in f:
            if v_idx not in head_vert_set:
                is_head = False
                break
        if is_head:
            head_ft = ft[f_idx]
            for vt_idx in head_ft:
                head_tex_verts_set.add(vt_idx )

    return np.array([v for v in head_tex_verts_set])

def warp_head_texture(org_uv_verts, mod_uv_verts, handle_idxs, triangles_in_head):
    V = igl.eigen.MatrixXd()
    V_bc = igl.eigen.MatrixXd()
    U_bc = igl.eigen.MatrixXd()

    F = igl.eigen.MatrixXi()
    b = igl.eigen.MatrixXi()

    n_bdr = len(handle_idxs)
    n_verts = org_uv_verts.shape[0]
    n_tris = len(triangles_in_head)

    F.resize(len(triangles_in_head), 3)
    for i in range(n_tris):
        assert len(triangles_in_head[i]) == 3
        for k in range(3):
            F[i, k] = triangles_in_head[i][k]

    V.resize(n_verts, 3)
    for i in range(n_verts):
        for k in range(2):
            V[i,k] = org_uv_verts[i, k]
        V[i, 2] = 0.0

    b.resize(n_bdr, 1)
    for i in range(n_bdr):
        b[i, 0] = handle_idxs[i]

    U_bc.resize(b.rows(), V.cols())
    V_bc.resize(b.rows(), V.cols())

    for i in range(n_bdr):
        for k in range(2):
            V_bc[i, k] = org_uv_verts[handle_idxs[i], k]
        V_bc[i, 2] = 0.0

    for i in range(n_bdr):
        for k in range(2):
            U_bc[i,k] = mod_uv_verts[handle_idxs[i], k]
        U_bc[i, 2] = 0.0

    D = igl.eigen.MatrixXd()
    D_bc = U_bc - V_bc
    print(D_bc)
    igl.harmonic(V, F, b, D_bc, 2, D)
    U = V + D

    verts_1 = np.copy(org_uv_verts)
    for i in range(len(org_uv_verts)):
        for k in range(2):
            verts_1[i, k] = U[i,k]

    return verts_1

from collections import defaultdict
def build_v_vt_map(faces, faces_tex):
    v_vt_map = defaultdict(set)
    for f, ft in zip(faces, faces_tex):
        for v, vt in zip(f, ft):
            v_vt_map[v].add(vt)

    result = []
    n = len(v_vt_map.keys())
    for i in range(n):
        result.append([vt for vt in v_vt_map[i]])

    return result

def faces_from_verts(all_faces, verts):
    vset = set(verts)
    faces = []
    for f in all_faces:
        in_group = True
        for v in f:
            if v not in vset:
                in_group = False
                break

        if in_group:
            faces.append(f)

    return faces

import os
if __name__ == '__main__':
    dir = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/'

    N_Facial_LD = 68

    mpath = os.path.join(*[dir, 'victoria_template_textured.obj'])
    out_mpath = os.path.join(*[dir, 'victoria_template_textured_warped.obj'])

    mesh = import_mesh_tex_obj(mpath)
    verts= mesh['v']
    verts_tex = mesh['vt']
    faces = mesh['f']
    faces_tex = mesh['ft']

    vgrp_path = os.path.join(*[dir, 'victoria_part_vert_idxs.pkl'])
    with open(vgrp_path, 'rb') as file:
        vgrp = pickle.load(file)
        vthead = find_head_tex_verts(mesh, vgrp['head_texture'])

    ld_idxs_path = os.path.join(*[dir, 'victoria_face_landmarks.pkl'])
    with open(ld_idxs_path, 'rb') as file:
        vic_facial_ld_idxs_dict = pickle.load(file)
        vic_facial_ld_v_idxs = []
        for i in range(N_Facial_LD):
            vic_facial_ld_v_idxs.append(vic_facial_ld_idxs_dict[i])
        assert len(set(vic_facial_ld_v_idxs)) == len(vic_facial_ld_v_idxs)

    v_vt_map = build_v_vt_map(faces, faces_tex)
    vt_face_ld = []
    for i in range(N_Facial_LD):
        uv_tex_idxs = v_vt_map[vic_facial_ld_v_idxs[i]]
        assert len(uv_tex_idxs) == 1
        vt_face_ld.append(uv_tex_idxs[0])

    vtbody_vthead_map  = dict((vbody, vhead) for vhead, vbody in enumerate(vthead))
    faces_tex_in_head  = faces_from_verts(faces_tex, vthead)
    vt_face_ld_in_head = [vtbody_vthead_map[vbody] for vbody in vt_face_ld]

    #debug. bring it the same scale of the prn texture
    #verts_tex -= 0.5
    #verts_tex += 0.5
    verts_tex *= 255.0

    vthead_co   = verts_tex[vthead]
    vt_face_ld_co  = verts_tex[vt_face_ld]

    #load target landmarks
    dir_1 = '/home/khanhhh/data_1/projects/Oh/codes/samples/PRNet/Data/uv-data/'
    tar_ld_path = os.path.join(*[dir_1, 'uv_kpt_ind.txt'])
    tar_face_ld_cos = np.loadtxt(tar_ld_path).T
    tar_face_ld_cos[:,1] = 255 - tar_face_ld_cos[:,1]
    tar_face_ld_cos = tar_face_ld_cos.astype(np.float32)

    sources = vt_face_ld_co
    targets = tar_face_ld_cos
    sources = sources.astype(np.float32)
    targets = targets.astype(np.float32)
    sources = sources.reshape((1, -1, 2))
    targets = targets.reshape((1, -1, 2))

    matches = list()
    for i in range(N_Facial_LD):
        matches.append(cv.DMatch(i, i, 0))
    tps = cv.createThinPlateSplineShapeTransformer(10)
    tps.estimateTransformation(sources, targets, matches)

    verts_tex_1 = verts_tex.copy().reshape((1, -1, 2)).astype(np.float32)
    error, verts_tex_1 = tps.applyTransformation(verts_tex_1)
    verts_tex_1.shape = (-1, 2)

    verts_tex_1 /= 255.0
    tar_face_ld_cos /= 255.0

    #verts_tex_1[verts_tex_1 < 0.0000001] = 0.0
    #verts_tex_1[verts_tex_1 > 1.0000001] = 0.0'
    min_uv = verts_tex_1.min()
    max_uv = verts_tex_1.max()
    range_uv = max_uv - min_uv
    verts_tex_1 = (verts_tex_1 - min_uv)/range_uv

    prn_rect_center = np.array([0.5,0.5])
    prn_rect_center = (prn_rect_center - min_uv) / range_uv
    prn_rect_size   = np.array([1.0, 1.0])/range_uv

    #export texture
    mesh['vt'] = verts_tex_1

    vt_face_ld_co_1  = verts_tex_1[vt_face_ld]
    export_mesh_tex_obj(fpath=out_mpath, mesh=mesh)

    path = np.savetxt(fname=os.path.join(*[dir, 'prn_texture_rectangle.txt']), X = np.vstack([prn_rect_center, prn_rect_size]))


    plt.axes().set_aspect(1.0)
    plt.plot(verts_tex_1[:,0], verts_tex_1[:,1], 'g+')
    plt.plot(vt_face_ld_co_1[:, 0], vt_face_ld_co_1[:, 1], 'r+')
    #plt.plot(tar_face_ld_cos[:,0], tar_face_ld_cos[:,1], 'b+', linewidth=2, markersize=4 )
    plt.plot(prn_rect_center[0], prn_rect_center[1], 'b+', linewidth=2, markersize=7 )

    plt.show()