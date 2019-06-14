import numpy as np
from common.obj_util import import_mesh_obj, import_mesh_tex_obj, export_mesh
from common.transformations import affine_matrix_from_points
import pickle
import matplotlib.pyplot as plt
from scipy import spatial
import sys
sys.path.insert(0, '/home/khanhhh/data_1/sample_codes/libigl/python')
import pyigl as igl

def axis_blender_convert(verts):
    tmp = np.copy(verts[:, 2])
    verts[:,2] = verts[:,1]
    verts[:, 1] = -tmp
    return verts

def find_prn_ld_idx(verts, ld_verts):
    points = np.c_[verts[:,0].ravel(), verts[:,1].ravel(), verts[:,2]]
    tree = spatial.KDTree(points)

    dsts, idxs = tree.query(ld_verts, k =1)
    for dst in dsts:
        assert dst < 0.00001, "incorrect vertex idx"
    return idxs

def deform_prn_head(org_head_verts, embedded_head_verts, handle_idxs, head_tris_in_head):
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

def calc_vert_index_in_face_space(vic_part_vert_idxs, vic_face_vert_idxs):
    vic_face_vert_idxs_map = dict([(vic_face_vert_idxs[i], i) for i in range(len(vic_face_vert_idxs))])

    for idx in vic_part_vert_idxs:
        assert idx in vic_face_vert_idxs_map, f'vertex {idx} does not blong to head'

    vic_part_vert_idxs_in_face = np.array([vic_face_vert_idxs_map[idx] for idx in vic_part_vert_idxs])

    return vic_part_vert_idxs_in_face

def calc_face_tris_in_face_space(tpl_tris, vface_map):
    #inverse mapping: from global indx to local index
    vface_map_hash = dict([(vface_map[i], i) for i in range(len(vface_map))])
    face_tris = []
    for tri in tpl_tris:
        in_face = True
        for v_tri in tri:
            if v_tri not in vface_map_hash:
                in_face = False
                break
        if in_face:
            v_tri_in_head = [vface_map_hash[v_tri] for v_tri in tri]
            face_tris.append(v_tri_in_head)

    return face_tris


if __name__ == '__main__':
    meta_dir = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/'
    vic_path = f'{meta_dir}/victoria_template_tri_textured.obj'
    vic_ld_path = f'{meta_dir}/victoria_face_landmarks.pkl'
    vic_grp_path = f'{meta_dir}/victoria_part_vert_idxs.pkl'

    vic_mesh = import_mesh_tex_obj(vic_path)
    vic_verts = vic_mesh['v']
    vic_tris = vic_mesh['f']

    with open(vic_ld_path, 'rb') as file:
        vic_ld_idxs_dict = pickle.load(file)
        assert len(vic_ld_idxs_dict) == 68
        vic_ld_idxs = [vic_ld_idxs_dict[i] for i in range(68)]
        upper_lip = [48722, 48759, 49775, 23972, 23935]
        under_lip = [49145, 49120, 49791, 24333, 24358]
        mid_eye   = [46895, 49728, 22108, 49670, 49733]
        ld_ext_idxs = np.array([48255, 48263, 49746, 23476, 23468]+ upper_lip + under_lip + mid_eye)
        vic_ld_idxs = np.hstack([vic_ld_idxs, ld_ext_idxs])
        vic_lds     = vic_verts[vic_ld_idxs, :]

    with open(vic_grp_path, 'rb') as file:
        vic_vert_grps = pickle.load(file)
        vic_face_vmap = vic_vert_grps['face']
        vic_face_tris = calc_face_tris_in_face_space(vic_tris, vic_face_vmap)

    vic_ld_idxs_in_face = calc_vert_index_in_face_space(vic_ld_idxs, vic_face_vmap)

    prn_dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/face/test_vic_prn_alignment'
    prn_path = f'{prn_dir}/img_model_2.obj'
    prn_ld_path = f'{prn_dir}/img_model_2_kpt.txt'

    prn_mesh = import_mesh_tex_obj(prn_path)
    prn_verts = prn_mesh['v']
    prn_tris = prn_mesh['f']
    prn_lds_org = np.loadtxt(prn_ld_path)
    prn_ld_idxs = find_prn_ld_idx(prn_verts, prn_lds_org)
    upper_lip = [31607, 31162, 31166, 31170, 31629]
    under_lip = [34035, 34042, 34046, 34050, 34056]
    mid_eye = [10685, 7995, 10717, 11439, 9963]
    prn_ld_ext_idxs = np.array([40, 84, 126, 170, 206] + upper_lip + under_lip + mid_eye)
    prn_ld_idxs = np.hstack([prn_ld_idxs, prn_ld_ext_idxs])

    prn_lds = prn_verts[prn_ld_idxs, :]

    prn_center = np.mean(prn_verts, axis=0)

    prn_verts -= prn_center
    prn_lds -= prn_center
    s = 0.01
    prn_verts *= s
    prn_lds *= s

    prn_verts = axis_blender_convert(prn_verts)
    prn_lds = axis_blender_convert(prn_lds)

    #move the prn face a bit up to the z level of victoria face
    dz = 17.0
    prn_verts[:,2] += dz
    prn_lds[:,2]  += dz

    vic_lds = vic_lds.T
    prn_lds = prn_lds.T
    AT = affine_matrix_from_points(prn_lds, vic_lds, shear = False, scale=True)

    prn_verts_1 = np.hstack([prn_verts, np.ones((prn_verts.shape[0], 1))])
    prn_verts_1 = np.dot(AT, prn_verts_1.T)
    prn_verts_1 = prn_verts_1.T[:,:3]

    out_dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/face/test_vic_prn_alignment/'

    export_mesh(f'{out_dir}/align_prn_face.obj', verts=prn_verts_1, faces=prn_tris)
    np.savetxt(fname=f'{out_dir}/align_prn_face_ld.txt', X=prn_lds.T)

    vic_face_verts_org = vic_verts[vic_face_vmap]
    vic_face_verts_mod = vic_face_verts_org.copy()
    vic_face_verts_mod[vic_ld_idxs_in_face] = prn_verts_1[prn_ld_idxs]

    vic_face_verts_mod_df = deform_prn_head(vic_face_verts_org, vic_face_verts_mod, vic_ld_idxs_in_face, vic_face_tris)

    export_mesh(f'{out_dir}/align_vic_face_tri_mod.obj', verts=vic_face_verts_mod, faces=vic_face_tris)
    export_mesh(f'{out_dir}/align_vic_face_tri_close.obj', verts=vic_face_verts_mod_df, faces=vic_face_tris)

