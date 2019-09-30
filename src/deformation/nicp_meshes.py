from numpy import empty, dot
import numpy as np
import scipy.io
#from tools import IO,Util
#from tools.IO import write_ply,write_obj,read_obj
import os
#from tools import geom,mesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.neighbors import NearestNeighbors

import pdb
import random
#import Video
#from pcSfm import bfm_landmark_indices
#from pcSfm import robust_lm
#from pcSfm import get_vertex_normals
from sksparse.cholmod import cholesky_AAt
#from sksparse.test_cholmod import cholesky_AAt
import cv2
import os
from numpy import *
from math import sqrt
from common.obj_util import import_mesh_tex_obj, export_mesh
from common.transformations import affine_matrix_from_points
import sys
sys.path.insert(0, '../../third_parties/libigl/python/')
import pyigl as igl

def per_vertex_normal(verts,  tris):
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()

    n_verts = verts.shape[0]
    n_tris = len(tris)

    F.resize(n_tris, 3)
    for i in range(n_tris):
        assert len(tris[i]) == 3
        for k in range(3):
            F[i, k] = tris[i][k]

    V.resize(n_verts, 3)
    for i in range(n_verts):
        for k in range(3):
            V[i, k] = verts[i, k]

    VN = igl.eigen.MatrixXd()
    igl.per_vertex_normals(V, F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, VN)

    VN_out = np.zeros_like(verts)
    for i in range(n_verts):
        for k in range(3):
            VN_out[i, k] = VN[i,k]

    return VN_out

def spsolve_chol(sparse_X, dense_b):
    factor = cholesky_AAt(sparse_X.T)
    return factor(sparse_X.T.dot(dense_b)).toarray()


# robust_lm = np.array( [36, 39, 42, 45, 30, 48, 54])

#bfm_landmark_indices = np.array([2088, 5959, 10603, 14472, 8319 , 5781, 11070 , 19770 , 35341])
#right eye out, in, left eye in,out , nosetip, right lip, left lip , right ear, left ear

#bfm_landmark_indices = np.array([2088, 5959, 10603, 8319 , 5781, 11070])
bfm_landmark_indices = np.random.randint(0, 12500, 50)
lbreast = np.array([i for i in range(9261-2, 9261+2, 1)])
rbreast = np.array([i for i in range(9552-2, 9552+2, 1)])
lubreast = np.array([i for i in range(8456-2, 8456+2, 1)])
rubreast = np.array([i for i in range(8774-2, 8774+2, 1)])
hip = np.array([i for i in range(7591-2, 7591+2, 1)])
lhip = np.array([i for i in range(7105-2, 7105+2, 1)])
rhip = np.array([i for i in range(7130-2, 7130+2, 1)])
bfm_landmark_indices = np.hstack([bfm_landmark_indices, lbreast, rbreast, lubreast, rubreast, hip, lhip, rhip])

#seq = '/home/shubham/datasets/groundtruth/head_mesh/Kurth/'
#meshpath = seq + 'Kurth.ply'
#target = IO.read_ply(meshpath,return_normals=False,return_faces=False)
#print("Done reading target...")
#gt_lms_path = seq+ '/lms.txt'
#gt_lms = np.loadtxt(gt_lms_path)
#nL = gt_lms.shape[0]

debug_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/debug/registration/'
target_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/mean_male_ucsc.obj'
target_mesh = import_mesh_tex_obj(target_path)
target = target_mesh['v']
gt_lms = target[bfm_landmark_indices, :]
nL = gt_lms.shape[0]

#source mesh
src_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/mean_female_ucsc.obj'
src_mesh = import_mesh_tex_obj(src_path)
mesh = src_mesh['v']
meshfaces = np.array(src_mesh['f'])
#mesh_norms = src_mesh['vn']
#mesh,meshfaces = read_obj('mean.obj')
#tls =meshfaces-1
tls = meshfaces

#print("template mean mesh norm :",np.linalg.norm(mesh[2087, :] - mesh[14471,:]))
#print ("ground truth mesh norm: ", np.linalg.norm (gt_lms[0] - gt_lms [3]))
##scale up template mesh##
#scale = np.linalg.norm(mesh[2087, :] - mesh[14471,:]) / np.linalg.norm (gt_lms[0] - gt_lms [3])
scale = (mesh[:,2].max() - mesh[:,2].min())/(target[:,2].max() - target[:,2].min())
mesh  = mesh/scale

#R, t = rigid_transform_3D(mesh[bfm_landmark_indices,:], gt_lms)
A = affine_matrix_from_points(mesh[bfm_landmark_indices,:].T, gt_lms.T)
R = A[:3,:3]
t = A[:3,3]
mesh = dot(R, mesh.T) + t.reshape(3, 1)
mesh = mesh.T
nV = mesh.shape[0]

centre = np.mean(target,axis=0)
#scale_factor = (np.linalg.norm(mesh[2087, :] - mesh[14471, :])) *2

#mesh_norms = get_vertex_normals(mesh, tls)
mesh_norms = per_vertex_normal(mesh, tls)

t_orig = target.copy()
target = target - centre
#target = target / scale_factor
mesh = mesh - centre
#mesh = mesh / scale_factor
orig_mesh = mesh.copy()
gt_lms = gt_lms - centre
#gt_lms = gt_lms / scale_factor

nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
nbrs50 = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(target)

nV = mesh.shape[0]
n = nV
#
print("num verts: ", n)
l = []
for i in tls:
    s = np.sort(i)
    l.append(tuple(s[:2]))
    l.append(tuple(s[1:]))

edgeset = set(l)
e = len(edgeset)
print("num edges:", e)

M = sparse.lil_matrix((e, n), dtype=np.float32)

for i, t in enumerate(edgeset):
    M[i, t[0]] = -1
    M[i, t[1]] = 1

gamma = 1
G = np.diag([1, 1, 1, gamma]).astype(np.float32)

Es = sparse.kron(M, G)
print("Es shape:", Es.shape)

UL = gt_lms
# #orig, pretty decent: ##
# alphas = [50,35,25,20,15, 10,5]
# betas = [12,8, 4, 2, 0.5, 0,0]
alphas = [50,35, 20, 15, 7,   3, 2, 1, 0.6, 0.4]
betas =  [15,14, 3,  2,  0.5, 0, 0, 0, 0,   0]

alphas.extend(10*[0.1])
betas.extend(10*[0])

alphas.extend(10*[0.05])
betas.extend(10*[0])

alphas.extend(10*[0.02])
betas.extend(10*[0])

alphas.extend(10*[0.001])
betas.extend(10*[0])

alphas.extend(10*[0.0005])
betas.extend(10*[0])

alphas.extend(10*[0.0001])
betas.extend(10*[0])

#mouth_mask = np.loadtxt('./basel_mouth_rigging.lm').astype(np.int32)

for cnt,(alpha_stiffness,beta) in enumerate(zip(alphas,betas)):

    print("using alpha stiffness and beta(lm wt): ",alpha_stiffness,beta)

    iter_max = 5
    if(alpha_stiffness<25):
        iter_max = 3

    for iter_cnt in range(iter_max):

        #####DATA WEIGHTS########
        weights = np.ones([n,1])
        if(alpha_stiffness>25):
            weights*=0.5

        if(alpha_stiffness>15): #no normals used
            distances, indices = nbrs.kneighbors(mesh)
            print("median mesh distance:",np.median(distances))
            indices = indices.squeeze()

            matches = target[indices]
        else:
            indices = None

            distances = np.zeros(mesh.shape[0])
            matches = np.zeros_like(mesh)
            #mesh_norms = get_vertex_normals(mesh, tls)
            mesh_norms = per_vertex_normal(mesh, tls)
            dist50, inds50 = nbrs50.kneighbors(mesh)
            eps = 2.5e-3  #remember, rescaling system to 1

            for i in range(mesh.shape[0]):
                pt_normal = mesh_norms[i]
                corr50 = target[inds50[i]]
                ab = mesh[i] - corr50
                c = np.cross(ab,pt_normal)
                line_dists = np.linalg.norm(c,axis =1)
                # print(np.min(line_dists),np.mean(line_dists),np.max(line_dists))
                fltr = line_dists<eps
                # pdb.set_trace()
                if(np.sum(fltr)>0):
                    matches[i,:]  = np.mean(corr50[fltr],axis=0)
                distances[i] = np.linalg.norm(matches[i] - mesh[i])

        #d_thresh = (np.linalg.norm(mesh[2087, :3] - mesh[14471, :3])) / 4
        d_thresh = 0.001
        mesh = np.hstack((mesh, np.ones([n, 1])))

        mismatches = np.where(distances>d_thresh)[0]
        weights[mismatches] = 0
        #weights[mouth_mask] =0

        mesh_matches = np.where(weights > 0)[0]
#
        print("Setting up D and V...")
        B = sparse.lil_matrix((4 * e + n, 3), dtype=np.float32)
        DL = sparse.lil_matrix((nL, 4 * n), dtype=np.float32)
        V = sparse.lil_matrix((n, 4 * n), dtype=np.float32)

        B[4 * e: (4 * e + n), :] = weights * matches
        for i in range(n):
            # D[i,4*i:4*i+4] = weights[i]*mesh[i]
            V[i, 4 * i:4 * i + 4] = mesh[i]

        D = V.multiply(weights)

        lm_wtmat = beta * np.ones([DL.shape[0], 1])

        for i, lm in enumerate(bfm_landmark_indices):
            DL[i, 4 * lm:4 * lm + 4] = lm_wtmat[i] * mesh[lm]  ##BETA moved here !!

        D = D.tocsr()
        DL = DL.tocsr()
        A = sparse.csr_matrix(sparse.vstack([alpha_stiffness * Es, D, DL]))

        B = sparse.vstack((B, lm_wtmat * UL))  ##assuming typo in paper, beta should be weighing both ?
        print("B size after lms", B.shape)
        print("solving...")
        X = spsolve_chol(A, B)

        print("warping...")
        new_verts = V.dot(X)  ##X from spsolve_chol is Dense already

        if (iter_cnt == iter_max - 1):
            print("saving...")
            #new_mesh_path = seq + '/deformed_alpha_' + str(alpha_stiffness) + '_mesh.obj'
            new_mesh_path = debug_dir + '/deformed_alpha_' + str(alpha_stiffness) + '_mesh.obj'
            if (alpha_stiffness == 15):
                #new_mesh_path = seq + '/final_mesh.obj'
                new_mesh_path = debug_dir + '/final_mesh.obj'
            #vs = np.asarray(new_verts) * scale_factor + centre
            vs = np.asanyarray(new_verts) + centre
            if (alpha_stiffness <= 15):
                match_mesh = vs[mesh_matches]
                #targ_matches = (matches[np.where(weights > 0)[0]]) * scale_factor + centre
                targ_matches = (matches[np.where(weights > 0)[0]]) + centre
                #write_ply(seq + '/matched_alpha_' + str(alpha_stiffness) + '_mesh.ply', match_mesh, normals=None)
                #write_ply(seq + '/matched_alpha_' + str(alpha_stiffness) + '_mesh.ply', match_mesh, normals=None)
                #export_mesh(debug_dir + '/matched_alpha_' + str(alpha_stiffness) + '_target.ply', verts=targ_matches)

            #IO.write_obj(new_mesh_path, vs, tls + 1)
            export_mesh(new_mesh_path, vs, tls)
            # rgb = project_and_sample_textures(vs, tls, frames, K, Rs, ts, stream, vis=vis, inds=None)
            # IO.write_plymesh_with_texture(seq + '/textured_deformed_' + str(alpha_stiffness) + '.ply', vs, rgb, tls, None)

        mesh = new_verts


# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Object-centric tracking.')
#     parser.add_argument(
#         'path', metavar='path', nargs=1, type=str,
#         help='Path to the sequence for processing')
#     parser.add_argument(
#         '--target', dest='target', action='store', default=None,
#         help='path to downsampled,cleaned final ply')
#
#     # parser.add_argument(
#     #     '--source', dest='source', action='store', default='/home/shubham/BA/dlib/shape_predictor_68_face_landmarks.dat',
#     #     help='path to mesh_tformed')
#     # parser.add_argument(
#     #     '--lms', dest='lms', action='store', default='/home/shubham/BA/dlib/shape_predictor_68_face_landmarks.dat',
#     #     help='path to mesh_tformed')
#     parser.add_argument(
#         '--bfm', dest='bfm_path', action='store', default='/home/shubham/Downloads/Basel2009/01_MorphableModel.mat',
#         help='path to morphable model mat file')
#
#
#     args = parser.parse_args()
#     # rectify_mesh(args.path[0],dlib_path = args.dlib_tracker,bfm_path = args.bfm_path)
#     # exit()
#     nicp_fit(args.path[0], args.bfm_path,args.target)
