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
from common.transformations import rotation_matrix, unit_vector, angle_between_vectors, vector_product, affine_matrix_from_points
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

    head_embed  = HmHeadEmbedder(meta_dir)
    face_warp   = HmFaceWarp(meta_dir)

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
