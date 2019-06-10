from deploy.hm_shape_pred_pytorch_model import HmShapePredPytorchModel
from deploy.hm_shape_pred_model import HmShapePredModel
from deploy.hm_sil_pred_model import HmSilPredModel
from deploy.hm_face_warp import  HmFaceWarp
from pca.nn_util import crop_silhouette_pair
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from common.obj_util import import_mesh_obj, export_mesh, import_mesh_tex_obj, export_mesh_tex_obj
from common.transformations import rotation_matrix, unit_vector, angle_between_vectors, vector_product
import math
import pickle
from deformation.ffdt_deformation_lib import  TemplateMeshDeform

import sys
sys.path.insert(0, '/home/khanhhh/data_1/sample_codes/libigl/python')
import pyigl as igl

def align_face_to_mesh(face_verts, mesh_verts, vmap_face_to_mesh):
    ver_ldm_idx_0 = 13118
    ver_ldm_idx_1 = 13150

    hor_ldm_idx_0 = 10673
    hor_ldm_idx_1 = 4062

    face_ver_ldm_co_0 =  face_verts[ver_ldm_idx_0]
    face_ver_ldm_co_1 =  face_verts[ver_ldm_idx_1]
    mesh_ver_ldm_co_0 = mesh_verts[vmap_face_to_mesh[ver_ldm_idx_0], :]
    mesh_ver_ldm_co_1 = mesh_verts[vmap_face_to_mesh[ver_ldm_idx_1], :]

    face_hor_ldm_co_0 =  face_verts[hor_ldm_idx_0]
    face_hor_ldm_co_1 =  face_verts[hor_ldm_idx_1]
    mesh_hor_ldm_co_0 = mesh_verts[vmap_face_to_mesh[hor_ldm_idx_0], :]
    mesh_hor_ldm_co_1 = mesh_verts[vmap_face_to_mesh[hor_ldm_idx_1], :]


    tmp = np.copy(face_verts[:, 2])
    face_verts[:,2] = face_verts[:,1]
    face_verts[:, 1] = -tmp
    face_verts[:, 0] = -face_verts[:, 0]

    #scale = abs(np.linalg.norm(mesh_ver_ldm_co_0 - mesh_ver_ldm_co_1) / np.linalg.norm(face_ver_ldm_co_0 - face_ver_ldm_co_1))
    mesh_face = mesh_verts[vmap_face_to_mesh]
    scale = (mesh_face[:,2].max() - mesh_face[:,2].min()) / (face_verts[:,2].max() - face_verts[:,2].min())
    scale /= 1.1
    face_verts *= scale

    rot_mat = rotation_matrix(-0.02* math.pi, [0.0, 1.0, 0.0])
    face_verts = np.hstack((face_verts, np.ones((face_verts.shape[0], 1))))
    face_verts = face_verts @ rot_mat
    face_verts = face_verts[:, :3]

    face_verts[:,0] -= 0.005

    mesh_face_center = np.mean(mesh_face, axis=0)
    face_center = np.mean(face_verts, axis=0)
    print(face_center, mesh_face_center)
    mesh_verts[vmap_face_to_mesh] = face_verts - face_ver_ldm_co_0 +  mesh_ver_ldm_co_0

def embbed_neck_seam_to_tpl_head(ctm_verts, tpl_head_verts, vneck_seam_map, vneck_seam_map_in_head):
    tpl_head_verts[vneck_seam_map_in_head] = ctm_verts[vneck_seam_map]
    return tpl_head_verts

def embbed_face_to_tpl_head(tpl_head_verts, face_verts, vface_map_in_head, vleye_map_in_head, vreye_map_in_head, vupper_lip_map_in_head):
    #orientation face to match victoria face
    tmp = np.copy(face_verts[:, 2])
    face_verts[:,2] = face_verts[:,1]
    face_verts[:, 1] = -tmp
    face_verts[:, 0] = -face_verts[:, 0]

    #scale face to match victoria's face
    org_face_verts = tpl_head_verts[vface_map_in_head]
    scale = (org_face_verts[:,2].max() - org_face_verts[:,2].min()) / (face_verts[:,2].max() - face_verts[:,2].min())
    face_verts *= scale
    face_center = np.mean(face_verts, axis=0)
    org_face_center = np.mean(org_face_verts, axis=0)


    tpl_leye = np.mean(tpl_head_verts[vleye_map_in_head], axis=0)
    tpl_reye = np.mean(tpl_head_verts[vreye_map_in_head], axis=0)
    tpl_eye = 0.5*(tpl_leye + tpl_reye)
    tpl_upper_lip = np.mean(tpl_head_verts[vupper_lip_map_in_head], axis=0)
    tpl_ver_dir = unit_vector(tpl_eye - tpl_upper_lip)

    face_verts = face_verts - face_center + org_face_center
    tpl_head_verts[vface_map_in_head] = face_verts

    leye = np.mean(tpl_head_verts[vleye_map_in_head], axis=0)
    reye = np.mean(tpl_head_verts[vreye_map_in_head], axis=0)
    eye = 0.5*(leye + reye)
    upper_lip = np.mean(tpl_head_verts[vupper_lip_map_in_head], axis=0)
    ver_dir = unit_vector(eye - upper_lip)

    M = rotation_matrix(angle_between_vectors(ver_dir, tpl_ver_dir), vector_product(ver_dir, tpl_ver_dir))
    tpl_head_verts[vface_map_in_head] = np.dot(tpl_head_verts[vface_map_in_head] - org_face_center, M[:3, :3].T) + org_face_center

    return tpl_head_verts

def embed_tpl_head_to_cusomter(tpl_head_verts, ctm_verts, vhead_map, vneck_seam_map, vneck_seam_map_in_head):
    ctm_head_verts = ctm_verts[vhead_map]
    ctm_head_center = np.mean(ctm_head_verts, axis=0)

    scale = (ctm_head_verts[:,2].max() - ctm_head_verts[:,2].min()) / (tpl_head_verts[:, 2].max() - tpl_head_verts[:, 2].min())
    head_center = np.mean(tpl_head_verts, axis=0)
    tpl_head_verts -= head_center
    tpl_head_verts *= scale
    tpl_head_verts += ctm_head_center

    ctm_neck_seam_center = np.mean(ctm_verts[vneck_seam_map], axis=0)
    neck_seam_center = np.mean(tpl_head_verts[vneck_seam_map_in_head], axis=0)

    tpl_head_verts = tpl_head_verts - neck_seam_center + ctm_neck_seam_center

    return tpl_head_verts

def solve_head_discontinuity(org_head_verts, embedded_head_verts, handle_idxs, head_tris_in_head):
    V = igl.eigen.MatrixXd()
    V_bc = igl.eigen.MatrixXd()
    U_bc = igl.eigen.MatrixXd()

    F = igl.eigen.MatrixXi()
    b = igl.eigen.MatrixXi()

    n_bdr = len(handle_idxs)
    n_verts = org_head_verts.shape[0]
    n_tris = len(head_tris_in_head)

    F.resize(len(head_tris_in_head), 3)
    for i in range(n_tris):
        assert len(head_tris_in_head[i]) == 3
        for k in range(3):
            F[i, k] = head_tris_in_head[i][k]

    V.resize(n_verts, 3)
    for i in range(n_verts):
        for k in range(3):
            V[i,k] = org_head_verts[i,k]

    b.resize(n_bdr, 1)
    for i in range(n_bdr):
        b[i, 0] = handle_idxs[i]

    U_bc.resize(b.rows(), V.cols())
    V_bc.resize(b.rows(), V.cols())

    for i in range(n_bdr):
        for k in range(3):
            V_bc[i, k] = org_head_verts[handle_idxs[i], k]

    for i in range(n_bdr):
        for k in range(3):
            U_bc[i,k] = embedded_head_verts[handle_idxs[i], k]

    D = igl.eigen.MatrixXd()
    D_bc = U_bc - V_bc
    igl.harmonic(V, F, b, D_bc, 2, D)
    U = V + D

    verts_1 = np.copy(org_head_verts)
    for i in range(len(org_head_verts)):
        for k in range(3):
            verts_1[i, k] = U[i,k]

    return verts_1

#vic_part_vert_idxs: vertex index of a part of Victoria in Vicotoria space
#vic_head_vert_idxs: vertex index of the head of Victoria in Vicotoria space
def calc_vert_index_in_head_space(vic_part_vert_idxs, vic_head_vert_idxs):
    vic_head_vert_idxs_map = dict([(vic_head_vert_idxs[i], i) for i in range(len(vic_head_vert_idxs))])

    for idx in vic_part_vert_idxs:
        assert idx in vic_head_vert_idxs_map, f'vertex {idx} does not blong to head'

    vic_part_vert_idxs_in_head = np.array([vic_head_vert_idxs_map[idx] for idx in vic_part_vert_idxs])

    return vic_part_vert_idxs_in_head

def calc_head_tris_in_head_space(tpl_tris, vhead_map):
    vhead_map_hash = dict([(vhead_map[i], i) for i in range(len(vhead_map))])
    head_tris = []
    for tri in tpl_tris:
        in_head = True
        for v_tri in tri:
            if v_tri not in vhead_map_hash:
                in_head = False
                break
        if in_head:
            v_tri_in_head = [vhead_map_hash[v_tri] for v_tri in tri]
            head_tris.append(v_tri_in_head)

    return head_tris

def deform_face(face_path):
    tpl_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/face_victoria.obj'
    data_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/global_face_vic_prnet_parameterization.pkl'

    tpl_verts, tpl_faces = import_mesh_obj(fpath=tpl_path)

    mesh = import_mesh_tex_obj(fpath=face_path)
    ctl_df_verts = mesh['v']
    ctl_df_faces = mesh['f']

    mean = np.mean(ctl_df_verts, axis=0)
    ctl_df_verts = ctl_df_verts - mean
    ctl_df_verts *= 0.02

    for idx, tris in enumerate(ctl_df_faces):
        assert (len(tris) == 3), f'face {idx} with len of {len(tris)} is not a triangle'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        vert_UVWs = data['template_vert_UVW']
        vert_weights = data['template_vert_weight']
        vert_effect_idxs = data['template_vert_effect_idxs']

        deform = TemplateMeshDeform(effective_range=4, use_mean_rad=False)
        deform.set_meshes(ctl_verts=ctl_df_verts, ctl_tris=ctl_df_faces, tpl_verts=tpl_verts, tpl_faces=tpl_faces)
        deform.set_parameterization(vert_tri_UVWs=vert_UVWs, vert_tri_weights=vert_weights,
                                    vert_effect_tri_idxs=vert_effect_idxs)

        del vert_UVWs
        del vert_weights
        del vert_effect_idxs
        del data

    tpl_df_verts = deform.deform(ctl_df_verts)

    return tpl_df_verts, tpl_faces


from deploy.hm_head_embedder import HmHeadEmbedder
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-face",  type=str, required=True)
    ap.add_argument("-f_sil", type=str, required=True)
    ap.add_argument("-s_sil", type=str, required=True)
    ap.add_argument("-face_img", type=str, required=True)
    ap.add_argument("-height", type=float, required=True)
    ap.add_argument("-gender", type=float, default=0.0, required=False)
    args = ap.parse_args()

    face_res_path = args.face
    sil_f_path = args.f_sil
    sil_s_path = args.s_sil
    height = args.height
    gender = args.gender
    face_img_path = args.face_img

    name_id = Path(sil_f_path).stem

    meta_dir = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/'
    out_dir = '/home/khanhhh/data_1/projects/Oh/data/face/test_merge_face_body/'

    head_embed = HmHeadEmbedder(meta_dir)
    face_warp = HmFaceWarp(meta_dir)

    tpl_mesh_tri_path = f'{meta_dir}/align_victoria_tri.obj'
    tpl_mesh_path = f'{meta_dir}align_victoria.obj'

    sil_f = cv.imread(sil_f_path, cv.IMREAD_GRAYSCALE)
    sil_s = cv.imread(sil_s_path, cv.IMREAD_GRAYSCALE)
    face_img = cv.imread(face_img_path)

    tmp_out_mesh_path = f'/home/khanhhh/data_1/projects/Oh/data/face/test_merge_face_body/{name_id}_tmp_verts.npy'
    if not Path(tmp_out_mesh_path).exists():
        path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_384_256_ml_fml_1/models/joint/shape_model.jlb'
        model = HmShapePredModel(path)
        pred = model.predict(sil_f, sil_s, height, gender)[0]
        ctm_mesh_verts = pred.reshape(pred.shape[0] // 3, 3)
        print(f'output mesh vertex shape: {ctm_mesh_verts.shape}')
        np.save(tmp_out_mesh_path, ctm_mesh_verts)
    else:
        ctm_mesh_verts = np.load(tmp_out_mesh_path)

    mesh = import_mesh_tex_obj(fpath=face_res_path)
    ctl_df_verts = mesh['v']

    ctm_mesh_verts = head_embed.embed(customer_df_verts=ctm_mesh_verts, prn_facelib_verts=ctl_df_verts)

    texture = face_warp.warp(face_img)

    text_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_template_textured.obj'
    tex_mesh = import_mesh_tex_obj(text_mesh_path)

    out_mesh = {'v':ctm_mesh_verts, 'vt':tex_mesh['vt'], 'f':tex_mesh['f'], 'ft':tex_mesh['ft']}

    export_mesh_tex_obj(os.path.join(*[out_dir, f'{name_id}_victoria_head_tex.obj']), out_mesh, img_tex=texture)

def backup():
    ap = argparse.ArgumentParser()
    ap.add_argument("-face",  type=str, required=True)
    ap.add_argument("-f_sil", type=str, required=True)
    ap.add_argument("-s_sil", type=str, required=True)
    ap.add_argument("-f_img", type=str, required=True)
    ap.add_argument("-height", type=float, required=True)
    ap.add_argument("-gender", type=float, default=0.0, required=False)
    args = ap.parse_args()

    face_res_path = args.face
    sil_f_path = args.f_sil
    sil_s_path = args.s_sil
    height = args.height
    gender = args.gender

    name_id = Path(sil_f_path).stem

    out_dir = '/home/khanhhh/data_1/projects/Oh/data/face/test_merge_face_body/'

    tpl_mesh_tri_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_victoria_tri.obj'
    tpl_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_victoria.obj'

    sil_f = cv.imread(sil_f_path, cv.IMREAD_GRAYSCALE)
    sil_s = cv.imread(sil_s_path, cv.IMREAD_GRAYSCALE)

    tmp_face_verts_path = f'/home/khanhhh/data_1/projects/Oh/data/face/test_merge_face_body/{name_id}_tmp_face_verts.npy'
    if not Path(tmp_face_verts_path).exists():
        face_verts, face_faces = deform_face(face_res_path)

        # y_forward = -y_forward
        face_verts[:, 1] = -face_verts[:, 1]

        np.save(tmp_face_verts_path, face_verts)
        export_mesh(os.path.join(*[out_dir, 'tmp_test_deformed_fae.obj']), verts=face_verts, faces=face_faces)
    else:
        face_verts = np.load(tmp_face_verts_path)

    tpl_verts, tpl_faces = import_mesh_obj(tpl_mesh_path)
    _, tpl_tris = import_mesh_obj(tpl_mesh_tri_path)

    tmp_out_mesh_path = f'/home/khanhhh/data_1/projects/Oh/data/face/test_merge_face_body/{name_id}_tmp_verts.npy'
    if not Path(tmp_out_mesh_path).exists():
        path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_384_256_ml_fml_1/models/joint/shape_model.jlb'
        model = HmShapePredModel(path)
        pred = model.predict(sil_f, sil_s, height, gender)[0]
        mesh_verts = pred.reshape(pred.shape[0]//3, 3)
        print(f'output mesh vertex shape: {mesh_verts.shape}')
        np.save(tmp_out_mesh_path, mesh_verts)
    else:
        mesh_verts = np.load(tmp_out_mesh_path)

    #template part vert indices
    part_vert_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_part_vert_idxs.pkl'
    with open(part_vert_path, 'rb') as file:
        vparts = pickle.load(file=file)
        vface_map = vparts['face']
        vneck_seam_map = vparts['neck_seam']
        vhead_map = vparts['head']
        vleye_map = vparts['leye']
        vreye_map = vparts['reye']
        vupper_lip_map = vparts['upper_lip']

        vface_map_in_head = calc_vert_index_in_head_space(vface_map, vhead_map)
        vneck_seam_map_in_head = calc_vert_index_in_head_space(vneck_seam_map, vhead_map)
        vleye_map_in_head = calc_vert_index_in_head_space(vleye_map, vhead_map)
        vreye_map_in_head = calc_vert_index_in_head_space(vreye_map, vhead_map)
        vupper_lip_map_in_head = calc_vert_index_in_head_space(vupper_lip_map, vhead_map)

        head_tri_in_head = calc_head_tris_in_head_space(tpl_tris, vhead_map)

    tpl_head_verts = tpl_verts[vhead_map]
    tpl_head_verts = embed_tpl_head_to_cusomter(tpl_head_verts=tpl_head_verts, ctm_verts=mesh_verts, vhead_map=vhead_map, vneck_seam_map=vneck_seam_map, vneck_seam_map_in_head=vneck_seam_map_in_head)

    new_head_verts = np.copy(tpl_head_verts)
    new_head_verts = embbed_face_to_tpl_head(tpl_head_verts=new_head_verts, face_verts=face_verts,
                                             vface_map_in_head=vface_map_in_head, vleye_map_in_head=vleye_map_in_head, vreye_map_in_head=vreye_map_in_head,
                                             vupper_lip_map_in_head=vupper_lip_map_in_head)

    new_head_verts = embbed_neck_seam_to_tpl_head(ctm_verts=mesh_verts, tpl_head_verts=new_head_verts, vneck_seam_map=vneck_seam_map, vneck_seam_map_in_head=vneck_seam_map_in_head)

    handle_idxs = np.hstack([vface_map_in_head, vneck_seam_map_in_head])
    new_head_verts_1 = solve_head_discontinuity(tpl_head_verts, new_head_verts, handle_idxs, head_tri_in_head)

    # mesh_verts_1 = np.copy(mesh_verts)
    # mesh_verts_1[vhead_map] = new_head_verts
    # export_mesh(os.path.join(*[out_dir, f'{name_id}_victoria_head_merged.obj']), verts=mesh_verts_1, faces=tpl_faces)

    mesh_verts_2 = np.copy(mesh_verts)
    mesh_verts_2[vhead_map] = new_head_verts_1
    #export_mesh(os.path.join(*[out_dir, f'{name_id}_victoria_head_merged_smoothed.obj']), verts=mesh_verts_2, faces=tpl_faces)

    #test texutured export
    text_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_template_textured.obj'
    tex_mesh = import_mesh_tex_obj(text_mesh_path)
    out_mesh = {'v':mesh_verts_2, 'vt':tex_mesh['vt'], 'f':tex_mesh['f'], 'ft':tex_mesh['ft']}

    export_mesh_tex_obj(os.path.join(*[out_dir, f'{name_id}_victoria_head_tex.obj']), out_mesh)