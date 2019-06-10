from common.obj_util import import_mesh_obj, import_mesh_tex_obj
from deformation.ffdt_deformation_lib import TemplateMeshDeform
import pickle
import numpy as np
from common.obj_util import import_mesh_obj, export_mesh, import_mesh_tex_obj, export_mesh_tex_obj
from common.transformations import rotation_matrix, unit_vector, angle_between_vectors, vector_product, affine_matrix_from_points
import math
import os
import pickle
from deformation.ffdt_deformation_lib import  TemplateMeshDeform

import sys
sys.path.insert(0, '/home/khanhhh/data_1/sample_codes/libigl/python')
import pyigl as igl

def embbed_neck_seam_to_tpl_head(ctm_verts, tpl_head_verts, vneck_seam_map, vneck_seam_map_in_head):
    tpl_head_verts[vneck_seam_map_in_head] = ctm_verts[vneck_seam_map]
    return tpl_head_verts

def embbed_face_to_tpl_head(tpl_head_verts, face_verts, vface_map_in_head):
    #orientation face to match victoria face
    tmp = np.copy(face_verts[:, 2])
    face_verts[:,2] = face_verts[:,1]
    face_verts[:, 1] = tmp
    face_verts[:, 0] = -face_verts[:, 0]

    #scale face to match victoria's face
    org_face_verts = tpl_head_verts[vface_map_in_head]
    scale = (org_face_verts[:,2].max() - org_face_verts[:,2].min()) / (face_verts[:,2].max() - face_verts[:,2].min())
    face_verts *= scale
    face_center = np.mean(face_verts, axis=0) #customer's face center
    org_face_center = np.mean(org_face_verts, axis=0) #victoria's face center

    match_idxs = [12170, 12184, 5559, 1142, 991, 13075, 7603, 9475, 13247, 6479, 13136, 13089]
    test_tpl_face_verts = tpl_head_verts[vface_map_in_head].copy()

    #temporary embed the customer face to extract keypoints
    face_verts = face_verts - face_center + org_face_center
    tpl_head_verts[vface_map_in_head] = face_verts

    set_0 = test_tpl_face_verts[match_idxs, :].T
    set_1 = face_verts[match_idxs, :].T

    AT = affine_matrix_from_points(set_1, set_0, shear = False, scale=False)

    tmp_ctm_faceverts = tpl_head_verts[vface_map_in_head]
    tmp_ctm_faceverts = np.hstack([tmp_ctm_faceverts, np.ones((tmp_ctm_faceverts.shape[0], 1))])

    tmp_ctm_faceverts = np.dot(AT, tmp_ctm_faceverts.T)
    tmp_ctm_faceverts = tmp_ctm_faceverts.T
    tmp_ctm_faceverts = tmp_ctm_faceverts[:, :3]

    tpl_head_verts[vface_map_in_head]  = tmp_ctm_faceverts

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

class HmHeadEmbedder:

    def __init__(self, meta_dir):
        vic_tpl_face_mesh_path = os.path.join(*[meta_dir, 'face_victoria.obj'])
        vic_tpl_mesh_tri_path = os.path.join(*[meta_dir, 'align_victoria_tri.obj'])
        prn_facelib_mesh_path = os.path.join(*[meta_dir, 'face_prn_facelib.obj'])
        parameterization_path = os.path.join(*[meta_dir, 'global_face_vic_prnet_parameterization.pkl'])
        vic_tpl_vertex_groups_path = os.path.join(*[meta_dir, 'victoria_part_vert_idxs.pkl'])


        prn_face_mesh = import_mesh_tex_obj(fpath=prn_facelib_mesh_path)
        ctl_df_verts = prn_face_mesh['v']
        ctl_df_faces = prn_face_mesh['f']

        for idx, tris in enumerate(ctl_df_faces):
            assert (len(tris) == 3), f'face {idx} with len of {len(tris)} is not a triangle'

        tpl_verts, tpl_faces = import_mesh_obj(fpath=vic_tpl_face_mesh_path)

        with open(parameterization_path, 'rb') as f:
            data = pickle.load(f)
            vert_UVWs = data['template_vert_UVW']
            vert_weights = data['template_vert_weight']
            vert_effect_idxs = data['template_vert_effect_idxs']

            self.deform = TemplateMeshDeform(effective_range=4, use_mean_rad=False)
            self.deform.set_meshes(ctl_verts=ctl_df_verts, ctl_tris=ctl_df_faces, tpl_verts=tpl_verts, tpl_faces=tpl_faces)
            self.deform.set_parameterization(vert_tri_UVWs=vert_UVWs, vert_tri_weights=vert_weights,
                                        vert_effect_tri_idxs=vert_effect_idxs)

            del vert_UVWs
            del vert_weights
            del vert_effect_idxs
            del data

        vic_tpl_verts, vic_tpl_tris =  import_mesh_obj(vic_tpl_mesh_tri_path)

        #head information
        with open(vic_tpl_vertex_groups_path, 'rb') as file:
            vparts = pickle.load(file=file)
            vface_map = vparts['face']
            self.vneck_seam_map = vparts['neck_seam']
            self.vhead_map = vparts['head']
            vleye_map = vparts['leye']
            vreye_map = vparts['reye']
            vupper_lip_map = vparts['upper_lip']

            self.vface_map_in_head = calc_vert_index_in_head_space(vface_map, self.vhead_map)
            self.vneck_seam_map_in_head = calc_vert_index_in_head_space(self.vneck_seam_map, self.vhead_map)
            self.vleye_map_in_head = calc_vert_index_in_head_space(vleye_map, self.vhead_map)
            self.vreye_map_in_head = calc_vert_index_in_head_space(vreye_map, self.vhead_map)
            self.vupper_lip_map_in_head = calc_vert_index_in_head_space(vupper_lip_map, self.vhead_map)

            self.head_tri_in_head = calc_head_tris_in_head_space(vic_tpl_tris, self.vhead_map)
            self.vic_tpl_head_verts = vic_tpl_verts[self.vhead_map]

    @property
    def vic_tpl_faces(self):
        return self.deform.tpl_faces

    def embed(self, customer_df_verts, prn_facelib_verts):
        mean = np.mean(prn_facelib_verts, axis=0)
        ctl_df_verts = prn_facelib_verts - mean
        ctl_df_verts *= 0.02

        ctm_face_verts = self.deform.deform(ctl_df_verts)
        # y_forward = -y_forward
        ctm_face_verts[:, 1] = -ctm_face_verts[:, 1]


        tpl_head_verts = self.vic_tpl_head_verts.copy()

        tpl_head_verts = embed_tpl_head_to_cusomter(tpl_head_verts=tpl_head_verts, ctm_verts=customer_df_verts,
                                                    vhead_map=self.vhead_map, vneck_seam_map=self.vneck_seam_map,
                                                    vneck_seam_map_in_head=self.vneck_seam_map_in_head)

        new_head_verts = np.copy(tpl_head_verts)
        new_head_verts = embbed_face_to_tpl_head(tpl_head_verts=new_head_verts, face_verts=ctm_face_verts,
                                                 vface_map_in_head=self.vface_map_in_head)

        new_head_verts = embbed_neck_seam_to_tpl_head(ctm_verts=customer_df_verts, tpl_head_verts=new_head_verts,
                                                      vneck_seam_map=self.vneck_seam_map,
                                                      vneck_seam_map_in_head=self.vneck_seam_map_in_head)

        handle_idxs = np.hstack([self.vface_map_in_head, self.vneck_seam_map_in_head])
        new_head_verts_1 = solve_head_discontinuity(tpl_head_verts, new_head_verts, handle_idxs, self.head_tri_in_head)
        #new_head_verts_1 = new_head_verts

        customer_df_verts[self.vhead_map] = new_head_verts_1

        return customer_df_verts
