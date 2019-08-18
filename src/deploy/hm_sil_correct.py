from sklearn.externals import joblib
import sys
sys.path.insert(0, '../../third_parties/libigl/python/')
import pyigl as igl
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import shutil
import argparse
from pathlib import Path
import pickle
import os
from pose.pose_extract_tfpose import PoseExtractorTf
from pose.pose_common import HumanPose
from pose.pose_common import CocoPart as HmPart
from common.transformations import angle_between_vectors
from pca.nn_util import crop_silhouette_pair
from deploy.hm_sil_pred_model import HmSilPredModel
from pathlib import Path
from  common.util import find_largest_contour, smooth_contour, resample_contour
import triangle as tr


class HmSilCorrector():
    def __init__(self, size = (384,256)):
        self.size = size
        pass

    def correct_f_sil(self, sil_f, pose=None):
        size = (384, 256)
        sil, _, trans_f, _ = crop_silhouette_pair(sil_f, None, sil_f, None, target_h=size[0], target_w=size[1],
                                                  px_height=int(0.9 * size[0]))
        sil = (sil > 0).astype(np.uint8) * 255

        if pose is None:
            return sil

        pose.append_img_transform(HumanPose.build_img_transform(img_w=img_pose.shape[1], img_h = img_pose.shape[0]))
        pose.append_img_transform(trans_f)

        lankle = pose.point(HmPart.LAnkle)
        rankle = pose.point(HmPart.RAnkle)
        lwrist = pose.point(HmPart.LWrist)
        rwrist = pose.point(HmPart.RWrist)

        target_lankle = np.array((109, 340))
        target_rankle = np.array((148, 340))
        target_lwrist = np.array((50,  180))
        target_rwrist = np.array((208, 180))

        contour = find_largest_contour(sil)
        # contour[:, 0, 0], contour[:, 0, 1] = smooth_contour(contour[:,0,0], contour[:,0,1])
        X, Y = resample_contour(contour[:, 0, 0], contour[:, 0, 1], 400)
        contour = np.vstack([X, Y]).T

        N = contour.shape[0]
        contour_edges = []
        for i in range(N):
            contour_edges.append((i, (i + 1) % N))
        contour_edges = np.array(contour_edges)

        points = np.vstack([contour, lankle, rankle, lwrist, rwrist])
        A = dict(vertices=points, segments = contour_edges, holes=[[0.0, 0.0],[0.0,0.0]])
        B = tr.triangulate(A, 'qpa')
        # verts, triangles = self.remove_outside_triangles(B['vertices'], B['triangles'], sil)
        #B['vertices'] = verts
        #B['triangles'] = triangles
        #tr.compare(plt, A, B)
        #plt.show()

        verts = B['vertices']
        triangles = B['triangles']

        lankle_idx = np.argmin(np.linalg.norm(verts - lankle, axis=1))
        rankle_idx = np.argmin(np.linalg.norm(verts - rankle, axis=1))
        lwrist_idx = np.argmin(np.linalg.norm(verts - lwrist, axis=1))
        rwrist_idx = np.argmin(np.linalg.norm(verts - rwrist, axis=1))

        target_ankle_len = np.linalg.norm(target_lankle - target_rankle)
        ankle_dir = np.array([1.0, 0.0]) if lankle[0] < rankle[0] else np.array([-1.0, 0.0])
        midankle = 0.5*(lankle + rankle)

        lankle_1 = midankle - 0.5*target_ankle_len*ankle_dir
        rankle_1 = midankle + 0.5*target_ankle_len*ankle_dir

        target_wrist_len = np.linalg.norm(target_lwrist - target_rwrist)
        wrist_hor_dir = np.array([1.0, 0.0]) if lwrist[0] < rwrist[0] else np.array([-1.0, 0.0])
        mid_wrist = 0.5*(lwrist+rwrist)
        lwrist_1 = mid_wrist - 0.5*target_wrist_len*wrist_hor_dir
        rwrist_1 = mid_wrist + 0.5*target_wrist_len*wrist_hor_dir

        verts_1 = self.solve_biharmonic(verts, triangles, [lankle_idx, rankle_idx, lwrist_idx, rwrist_idx], np.vstack([lankle_1, rankle_1, lwrist_1, rwrist_1]))

        sil_corrected = np.zeros_like(sil)
        self.fill_silhouette(verts_1, B['triangles'], sil_corrected)

       # verts_1[:, 1] = sil.shape[0] - verts_1[:, 1]
       # A['vertices'][:, 1] = sil.shape[0] - A['vertices'][:, 1]

       # B['vertices'] = verts_1
       # B['triangles'] = triangles
       # tr.compare(plt, A, B)
       # plt.show()
        return sil_corrected

    def correct_s_sil(self, sil_s, pose):
        size = (384, 256)
        _, sil, _, trans_s = crop_silhouette_pair(None, sil_s, None, sil_s, target_h=size[0], target_w=size[1],
                                                  px_height=int(0.9 * size[0]))
        sil = (sil > 0).astype(np.uint8) * 255
        return sil

    @staticmethod
    def fill_silhouette(verts, triangles, sil_img):
        for i in range(triangles.shape[0]):
            t = triangles[i,:]
            points = np.array([verts[idx,:] for idx in t])
            points = points.reshape(-1, 1, 2).astype(np.int32)
            cv.fillConvexPoly(sil_img, points, color=(255,255,255))

        plt.imshow(sil_img)
        plt.show()

    @staticmethod
    def remove_outside_triangles(verts, triangles, sil_mask):
        centers = 0.3333*(verts[triangles[:,0], :] + verts[triangles[:,1], :]+verts[triangles[:,2], :])
        centers = centers.astype(np.uint32)
        centers_inside = sil_mask[centers[:,1],centers[:,0]] > 0

        new_triangles = triangles[centers_inside, :]
        new_vert_idxs = np.unique(new_triangles[:])
        new_verts = verts[new_vert_idxs, :]
        vmap = dict((new_vert_idxs[i], i) for i in range(new_vert_idxs.shape[0]))

        map_vidx_func = lambda old_idx : vmap[old_idx]
        new_triangles = np.vectorize(map_vidx_func)(new_triangles)

        return new_verts, new_triangles

    @staticmethod
    def solve_biharmonic(verts, triangles, handle_idxs, moved_handles):
        V = igl.eigen.MatrixXd()
        V_bc = igl.eigen.MatrixXd()
        U_bc = igl.eigen.MatrixXd()

        F = igl.eigen.MatrixXi()
        b = igl.eigen.MatrixXi()

        n_bdr = len(handle_idxs)
        n_verts = verts.shape[0]
        n_tris = triangles.shape[0]

        F.resize(n_tris, 3)
        for i in range(n_tris):
            for k in range(3):
                F[i, k] = triangles[i][k]

        V.resize(n_verts, 2)
        for i in range(n_verts):
            for k in range(2):
                V[i,k] = verts[i,k]

        b.resize(n_bdr, 1)
        for i in range(n_bdr):
            b[i, 0] = handle_idxs[i]

        U_bc.resize(b.rows(), V.cols())
        V_bc.resize(b.rows(), V.cols())

        for i in range(n_bdr):
            for k in range(2):
                V_bc[i, k] = verts[handle_idxs[i], k]

        for i in range(n_bdr):
            for k in range(2):
                U_bc[i,k] = moved_handles[i, k]

        D = igl.eigen.MatrixXd()
        D_bc = U_bc - V_bc
        igl.harmonic(V, F, b, D_bc, 2, D)
        U = V + D

        verts_1 = np.copy(verts)
        for i in range(n_verts):
            for k in range(2):
                verts_1[i, k] = U[i,k]

        return verts_1

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, help="image folder")
    ap.add_argument("-o", "--output_dir", required=True, help='output pose dir')
    ap.add_argument('-debug_name', required=False, type=str, default='')
    args = ap.parse_args()

   # test_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_384_256_ml_fml_nosyn/sil_f/test/'
   # for i, path in enumerate(Path(test_dir).glob('*.jpg')):
   #  img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
   #  plt.imshow(img, alpha=0.5)
   #  if i > 500:
   #      break
   # plt.show()

    debug_name = args.debug_name
   # sil_path_0 = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_384_256_ml_fml_nosyn/sil_f/test/_female_CSR0014A.jpg'
   # #sil_path_1 = '/home/khanhhh/data_1/projects/Oh/data/3d_human/test_data/body_designer_result_nosyn/cory_1933_front_sil.jpg'
   # sil_path_1 = '/home/khanhhh/data_1/projects/Oh/data/3d_human/test_data/body_designer_result_nosyn/female_front_IMG_1002_sil.jpg'
   # sil0 = cv.imread(sil_path_0, cv.IMREAD_GRAYSCALE)
   # sil1 = cv.imread(sil_path_1, cv.IMREAD_GRAYSCALE)
   # size = (384, 256)
   # sil1, _, trans_f, _ = crop_silhouette_pair(sil1, None, sil1, None, target_h=size[0], target_w=size[1], px_height=int(0.9*size[0]))
    # plt.subplot(221)
    # plt.imshow(sil0)
    # plt.subplot(222)
    # plt.imshow(sil1)
    # plt.subplot(223)
    # plt.imshow(sil1)
    # plt.subplot(224)
    # plt.imshow(sil0)
    # plt.imshow(sil1, alpha=0.5)
    # plt.show()

    os.makedirs(args.output_dir, exist_ok=True)

    hmsil_model_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/deploy_models_nosyn/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'

    tmp_dir = '/home/khanhhh/data_1/projects/Oh/data/images/oh_team_imgs/tmp/'
    img_path = Path('/home/khanhhh/data_1/projects/Oh/data/images/oh_team_imgs/cory_1933_front.jpg')
    img = cv.imread(str(img_path))

    sil_tmp_path = f'{tmp_dir}/{img_path.stem}_sil.jpg'
    if Path(sil_tmp_path).exists():
        sil = cv.imread(sil_tmp_path, cv.IMREAD_GRAYSCALE)
    else:
        hmsil_model = HmSilPredModel(model_path=hmsil_model_path, use_gpu=True, use_mobile_model=False)
        sil = hmsil_model.extract_silhouette(img)
        cv.imwrite(sil_tmp_path, img=sil)

    pose_tmp_path = f'{tmp_dir}/{img_path.stem}_pose.jlb'
    if Path(pose_tmp_path).exists():
        data = joblib.load(pose_tmp_path)
        pose = data['pose']
        img_pose = data['img_pose']
    else:
        extractor = PoseExtractorTf()
        pose, img_pose = extractor.extract_single_pose(img, debug=True)
        joblib.dump(filename=pose_tmp_path, value={'pose':pose, 'img_pose':img_pose})

    sil_corrector = HmSilCorrector()
    sil_corrector.correct_f_sil(sil, pose)

    cv.imwrite(f'{args.output_dir}/{img_path.stem}.png', img_pose)
    print(f'export pose to file {img_path}')
    with open(f'{args.output_dir}/{img_path.stem}.pkl', 'wb') as file:
        pickle.dump(obj = pose, file=file)

