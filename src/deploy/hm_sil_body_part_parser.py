import numpy as np
import cv2 as cv
from pose.pose_common import CocoPart as PoseParts
import matplotlib.pyplot as plt
import argparse
import os

def closest_point_idx(contour, point):
    dsts = np.linalg.norm(contour - point, axis=1)
    min_idx = np.argmin(dsts)
    return min_idx

class HmSilColorBodyPartParser:

    def __init__(self):
        pass

    def parse_sil_f(self, sil_f, contour_f, pose_f):
        lshoulder_jnt = pose_f.point(PoseParts.LShoulder)
        rshoulder_jnt = pose_f.point(PoseParts.RShoulder)
        lshoulder_jnt[1] = rshoulder_jnt[1] = 0.5*(lshoulder_jnt[1] + rshoulder_jnt[1])
        neck_jnt = pose_f.point(PoseParts.Neck)

        lhip_jnt = pose_f.point(PoseParts.LHip)
        rhip_jnt = pose_f.point(PoseParts.RHip)
        lhip_jnt[1] = rhip_jnt[1] = 0.5*(lhip_jnt[1] + rhip_jnt[1])
        midhip = 0.5*(lhip_jnt + rhip_jnt)

        lwrist_jnt = pose_f.point(PoseParts.LWrist)
        rwrist_jnt = pose_f.point(PoseParts.RWrist)
        lknee_jnt = pose_f.point(PoseParts.LKnee)
        rknee_jnt = pose_f.point(PoseParts.RKnee)
        lankle_jnt = pose_f.point(PoseParts.LAnkle)
        rankle_jnt = pose_f.point(PoseParts.RAnkle)

        lshoulder_idx, rshoulder_idx = self.estimate_shoulder_contour_points(contour_f, lshoulder_jnt, rshoulder_jnt)
        larmpit_idx, rarmpit_idx = self.estimate_front_armpits(contour_f, lwrist_jnt, rwrist_jnt, lhip_jnt, rhip_jnt)
        lhip_idx, rhip_idx = self.estimate_hip_contour_points(contour_f, lhip_jnt, rhip_jnt)
        lneck_idx, rneck_idx = self.estimate_neck_contour_points(contour_f, neck_jnt)
        crotch_idx = self.estimate_crotch_contour_point(contour, lknee_jnt, rknee_jnt)
        lankle_0_idx, lankle_1_idx = self.estimate_ankle_contour_points(contour, lankle_jnt)
        rankle_0_idx, rankle_1_idx = self.estimate_ankle_contour_points(contour, rankle_jnt)


        img = np.zeros((sil_f.shape[0], sil_f.shape[1],3), dtype=np.uint8)

        torso_contour_segs = [contour[rneck_idx:rshoulder_idx+1, :], rshoulder_jnt.reshape(1,2), contour[rarmpit_idx:rhip_idx+1, :]]
        torso_contour_segs.extend([rhip_jnt.reshape(1,2), lhip_jnt.reshape(1,2)])
        torso_contour_segs.extend([contour[lhip_idx:larmpit_idx+1, :], lshoulder_jnt.reshape(1,2)])
        torso_contour_segs.extend([contour[lshoulder_idx:lneck_idx+1, :], contour[rneck_idx,:].reshape(1,2)])
        torso = np.vstack(torso_contour_segs).astype(np.int32)

        larm_segs = [contour[larmpit_idx:lshoulder_idx+1,:],
                     lshoulder_jnt.reshape(1,2),
                     contour[larmpit_idx,:].reshape(1,2)]
        larm = np.vstack(larm_segs).astype(np.int32)

        rarm_segs = [contour[rshoulder_idx:rarmpit_idx+1,:],
                     rshoulder_jnt.reshape(1,2),
                     contour[rshoulder_idx,:].reshape(1,2)]
        rarm = np.vstack(rarm_segs).astype(np.int32)

        head_segs = [contour[lneck_idx:, :], contour[0:rneck_idx+1,:]]
        head = np.vstack(head_segs).astype(np.int32)

        rleg_segs = [contour[rhip_idx:crotch_idx+1, :], midhip.reshape(1,2)]
        rleg = np.vstack(rleg_segs).astype(np.int32)
        lleg_segs = [contour[crotch_idx:lhip_idx+1, :], midhip.reshape(1,2)]
        lleg = np.vstack(lleg_segs).astype(np.int32)


        cv.fillConvexPoly(img, torso.reshape(-1, 1, 2), (255,0,0))
        cv.fillConvexPoly(img, larm.reshape(-1, 1, 2), (0,255,0))
        cv.fillConvexPoly(img, rarm.reshape(-1, 1, 2), (0,255,0))
        cv.fillConvexPoly(img, head.reshape(-1, 1, 2), (0,0,255))
        cv.fillConvexPoly(img, rleg.reshape(-1, 1, 2), (0,255,255))
        cv.fillConvexPoly(img, lleg.reshape(-1, 1, 2), (255,255,0))

        if lankle_0_idx >=0 and lankle_1_idx >=0 and lankle_0_idx <  lankle_1_idx:
            lfeet = contour[lankle_0_idx:lankle_1_idx, :].astype(np.int32)
            cv.fillConvexPoly(img, lfeet.reshape(-1, 1, 2), (255,0,0))

        if rankle_0_idx >=0 and rankle_1_idx >=0 and rankle_0_idx <  rankle_1_idx:
            rfeet = contour[rankle_0_idx:rankle_1_idx, :].astype(np.int32)
            cv.fillConvexPoly(img, rfeet.reshape(-1, 1, 2), (255,0,0))

        plt.imshow(img)
        plt.plot(contour_f[:,0], contour_f[:,1], '-b')
        #plt.plot(torso[:,0], torso[:,1], '-b')
        plt.show()


    @staticmethod
    def plot_points(ax, points, mode = "+r"):
        for p in points:
            ax.plot(p[0], p[1], mode)

    @staticmethod
    def estimate_crotch_contour_point(contour, lknee, rknee):
        start_idx = closest_point_idx(contour, rknee)
        end_idx = closest_point_idx(contour, lknee)
        points = contour[start_idx:end_idx, :]
        crotch_local_idx = np.argmin(points[:,1])
        crotch = points[crotch_local_idx, :]
        crotch_idx = closest_point_idx(contour, crotch)
        return crotch_idx

    @staticmethod
    def estimate_ankle_contour_points(contour, ankle_jnt):
        points_0_mask = contour[:,0] < ankle_jnt[0]
        points_0 = contour[points_0_mask, :]
        if points_0.shape[0] > 0:
            dsts_0 = np.linalg.norm(points_0 - ankle_jnt, axis=1)
            ankle_0 = points_0[np.argmin(dsts_0), :]
            ankle_0_idx = closest_point_idx(contour, ankle_0)
        else:
            ankle_0_idx = -1

        points_1 = contour[np.bitwise_not(points_0_mask), :]
        if points_1.shape[0] > 0:
            dsts_1 = np.linalg.norm(points_1-ankle_jnt, axis=1)
            ankle_1 = points_1[np.argmin(dsts_1), :]
            ankle_1_idx = closest_point_idx(contour, ankle_1)
        else:
            ankle_1_idx = -1

        return ankle_0_idx, ankle_1_idx

    @staticmethod
    def estimate_neck_contour_points(contour, neck_jnt):

        lpoints_mask = np.bitwise_and(contour[:,0] > neck_jnt[0], contour[:,1] < neck_jnt[1])
        lpoints = contour[lpoints_mask, :]

        ldiag_dir = np.array([1.0,-1.0]); ldiag_dir = ldiag_dir/np.linalg.norm(ldiag_dir)
        lproj_dsts = np.dot(lpoints-neck_jnt, ldiag_dir)
        lshoulder_local_idx = np.argmin(lproj_dsts)
        lshoulder = lpoints[lshoulder_local_idx, :]
        #TODO: find a better way to find the index of the contour point: rshoulder
        lshoulder_idx = closest_point_idx(contour, lshoulder)

        #plt.axes().set_aspect(1.0)
        #plt.plot(contour[:,0], contour[:,1], '-b')
        #plt.plot(lpoints[:,0], lpoints[:,1], '+r')
        #plt.plot(lshoulder_joint[0], lshoulder_joint[1], '+r')
        #plt.plot(lshoulder[0], lshoulder[1], '+r')
        #plt.show()

        ###################
        rpoints_mask = np.bitwise_and(contour[:,0] < neck_jnt[0], contour[:,1] < neck_jnt[1])
        rpoints = contour[rpoints_mask, :]

        rdiag_dir = np.array([-1.0, -1.0]); rdiag_dir = rdiag_dir/np.linalg.norm(rdiag_dir)
        proj_dsts = np.dot(rpoints-neck_jnt, rdiag_dir)
        rshoulder_local_idx = np.argmin(proj_dsts)
        rshoulder = rpoints[rshoulder_local_idx,:]
        #TODO: find a better way to find the index of the contour point: rshoulder
        rshoulder_idx = closest_point_idx(contour, rshoulder)
        return lshoulder_idx, rshoulder_idx

    @staticmethod
    def estimate_hip_contour_points(contour, lhip_jnt, rhip_jnt):
        assert rhip_jnt[0] < lhip_jnt[0], 'incorrect left and right hip oder'

        lpoints_mask = contour[:,0] > lhip_jnt[0]
        lpoints = contour[lpoints_mask, :]
        l_dsts = np.linalg.norm(lpoints - lhip_jnt, axis=1)
        lhip_local_idx = np.argmin(l_dsts)
        lhip = lpoints[lhip_local_idx, :]
        lhip_idx = closest_point_idx(contour, lhip)

        rpoints_mask = contour[:, 0] < rhip_jnt[0]
        rpoints = contour[rpoints_mask, :]
        r_dsts = np.linalg.norm(rpoints - rhip_jnt, axis=1)
        rhip_local_idx = np.argmin(r_dsts)
        rhip = rpoints[rhip_local_idx, :]
        rhip_idx = closest_point_idx(contour, rhip)

        return lhip_idx, rhip_idx

    @staticmethod
    def estimate_shoulder_contour_points(contour, lshoulder_joint, rshoulder_joint):

        assert rshoulder_joint[0] < lshoulder_joint[0], 'incorrect left and right shoulder oder'

        lpoints_mask = np.bitwise_and(contour[:,0] > lshoulder_joint[0], contour[:,1] < lshoulder_joint[1])
        lpoints = contour[lpoints_mask, :]

        ldiag_dir = np.array([1.0,-1.0]); ldiag_dir = ldiag_dir/np.linalg.norm(ldiag_dir)
        lproj_dsts = np.dot(lpoints-lshoulder_joint, ldiag_dir)
        lshoulder_local_idx = np.argmax(lproj_dsts)
        lshoulder = lpoints[lshoulder_local_idx, :]
        #TODO: find a better way to find the index of the contour point: rshoulder
        lshoulder_idx = closest_point_idx(contour, lshoulder)

        #plt.axes().set_aspect(1.0)
        #plt.plot(contour[:,0], contour[:,1], '-b')
        #plt.plot(lpoints[:,0], lpoints[:,1], '+r')
        #plt.plot(lshoulder_joint[0], lshoulder_joint[1], '+r')
        #plt.plot(lshoulder[0], lshoulder[1], '+r')
        #plt.show()

        ###################
        rpoints_mask = np.bitwise_and(contour[:,0] < rshoulder_joint[0], contour[:,1] < rshoulder_joint[1])
        rpoints = contour[rpoints_mask, :]

        rdiag_dir = np.array([-1.0, -1.0]); rdiag_dir = rdiag_dir/np.linalg.norm(rdiag_dir)
        proj_dsts = np.dot(rpoints-rshoulder_joint, rdiag_dir)
        rshoulder_local_idx = np.argmax(proj_dsts)
        rshoulder = rpoints[rshoulder_local_idx,:]
        #TODO: find a better way to find the index of the contour point: rshoulder
        rshoulder_idx = closest_point_idx(contour, rshoulder)
        return lshoulder_idx, rshoulder_idx

    @staticmethod
    def estimate_front_armpits(contour_f, lwrist, rwrist, lhip, rhip):
        assert rhip[0] > rwrist[0]
        rwrist_idx = closest_point_idx(contour_f, rwrist)
        rhip_idx = closest_point_idx(contour_f, rhip)
        if rwrist_idx > rhip_idx:
            raise Exception('estimate_front_armpit_1: incorrect relative order of rwirst and rhip. it might happen due to contour orientation')

        #find the highest point
        rchain = np.array(contour_f[rwrist_idx:rhip_idx, :])
        rarmpit = rchain[np.argmin(rchain[:,1]),:]
        rarmpit_idx = closest_point_idx(contour, rarmpit)

        assert lhip[0] < lwrist[0]
        lwrist_idx = closest_point_idx(contour_f, lwrist)
        lhip_idx = closest_point_idx(contour_f, lhip)
        if lwrist_idx < lhip_idx:
            raise Exception('estimate_front_armpit_1: incorrect relative order of rwirst and rhip. it might happen due to contour orientation')

        #find the highest point
        lchain = np.array(contour_f[lhip_idx:lwrist_idx, :])
        larmpit = lchain[np.argmin(lchain[:,1]),:]
        larmpit_idx = closest_point_idx(contour, larmpit)
        return larmpit_idx, rarmpit_idx

from pathlib import Path
from sklearn.externals import joblib
from pose.pose_extract_tfpose import PoseExtractorTf
from pose.pose_common import HumanPose
from deploy.hm_sil_pred_model import HmSilPredModel
import pickle
from  common.util import find_largest_contour, smooth_contour, resample_contour

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, help="image folder")
    ap.add_argument("-o", "--output_dir", required=True, help='output pose dir')
    ap.add_argument('-debug_name', required=False, type=str, default='')
    ap.add_argument('-c', "--cache", action='store_true')
    args = ap.parse_args()


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

    tmp_dir = '/home/khanhhh/data_1/projects/Oh/data/images/oh_team_imgs/tmp_sil_parser/'
    os.makedirs(tmp_dir, exist_ok=True)
    img_path = Path('/home/khanhhh/data_1/projects/Oh/data/images/oh_team_imgs/cory_1933_front.jpg')
    #img_path = Path('/home/khanhhh/data_1/projects/Oh/data/images/oh_team_imgs/designer_2_side.jpg')
    img = cv.imread(str(img_path))

    sil_tmp_path = f'{tmp_dir}/{img_path.stem}_sil.jpg'
    if Path(sil_tmp_path).exists() and args.cache:
        sil = cv.imread(sil_tmp_path, cv.IMREAD_GRAYSCALE)
    else:
        hmsil_model = HmSilPredModel(model_path=hmsil_model_path, use_gpu=True, use_mobile_model=False)
        sil = hmsil_model.extract_silhouette(img)
        if args.cache:
            cv.imwrite(sil_tmp_path, img=sil)

    pose_tmp_path = f'{tmp_dir}/{img_path.stem}_pose.jlb'
    if Path(pose_tmp_path).exists() and args.cache:
        data = joblib.load(pose_tmp_path)
        pose = data['pose']
        img_pose = data['img_pose']
    else:
        extractor = PoseExtractorTf()
        pose, img_pose = extractor.extract_single_pose(img[:,:,::-1], debug=True)
        pose.append_img_transform(HumanPose.build_img_transform(img_w=img.shape[1], img_h=img.shape[0]))
        #plt.imshow(img_pose)
        #plt.show()
        if args.cache:
            joblib.dump(filename=pose_tmp_path, value={'pose':pose, 'img_pose':img_pose})

    sil_corrector = HmSilColorBodyPartParser()
    contour = find_largest_contour(sil, app_type=cv.CHAIN_APPROX_NONE)
    X, Y = resample_contour(contour[:, 0, 0], contour[:, 0, 1], 700)
    contour = np.vstack([X, Y]).T
    sil_corrector.parse_sil_f(sil, contour,pose)

    cv.imwrite(f'{args.output_dir}/{img_path.stem}.png', img_pose)
    print(f'export pose to file {img_path}')
    with open(f'{args.output_dir}/{img_path.stem}.pkl', 'wb') as file:
        pickle.dump(obj = pose, file=file)
