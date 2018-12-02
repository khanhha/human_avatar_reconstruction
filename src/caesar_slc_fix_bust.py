import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import os
import sys
from shapely.geometry import LinearRing, LineString, Point, MultiPoint
import shapely.geometry as geo
import shapely.ops as ops
import math
import shutil
import src.util as util
from numpy.linalg import norm
from copy import copy

def roll_contour(contour, shift):
    X =  np.roll(contour[:,0], shift)
    Y =  np.roll(contour[:,1], shift)
    contour = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    return contour

def preprocess_contour(contour, resolution = 400):
    X, Y = util.smooth_contour(contour[:, 0], contour[:, 1], sigma=2)
    X, Y = util.resample_contour(X, Y, resolution)

    ymax_idx = np.argmax(Y)

    X =  np.roll(X, -ymax_idx)
    Y =  np.roll(Y, -ymax_idx)

    #clockwise order
    p0 = np.array([X[0], Y[0]])
    pnext = np.array([X[7], Y[7]])
    if (pnext - p0).dot(np.array([1.0, 0])) < 0:
        X = X[::-1]
        Y = Y[::-1]

    contour = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

    return contour

def arm_inside_contour(contour_str, p):
    if p[0] > 0.0:
        p1 = p + 2.0*np.array([1, 0])
    else:
        p1 = p + 2.0*np.array([-1, 0])

    isct_p = LineString([p, p1]).intersection(contour_str)
    if isct_p.type == 'Point':
        return True
    else:
        return False

def extract_contour_chain(contour, x_0, x_1):
    n_point = contour.shape[0]
    chains = []
    chain = []
    for i in range(n_point):
        if x_0 < contour[i,0] and contour[i,0] < x_1:
            chain.append(i)
        else:
            if len(chain) > 0:
                chains.append(chain)
                chain = []
    return chains

def closest_point_idx(contour, point):
    diffs = contour - point
    dists = np.sum(np.square(diffs), axis=1)
    return np.argmin(dists)

#contour: clockwise order
#landmarks: contain points inside armt parts
def remove_arm_from_under_bust_slice(contour, arm_pnt_negx, arm_pnt_posx, debug_path = None):

    contour = preprocess_contour(contour)

    hor_dir = arm_pnt_posx - arm_pnt_negx

    if debug_path is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(contour[:,0], contour[:,1], '-b')
        plt.plot(arm_pnt_negx[0], arm_pnt_negx[1], '+r')
        plt.plot(arm_pnt_posx[0], arm_pnt_posx[1], '+r')

    fixed_left = False
    fixed_right = False
    has_left_arm = False
    has_right_arm = False

    contour_str = LinearRing([contour[i,:] for i in range(contour.shape[0])])
    if arm_inside_contour(contour_str, arm_pnt_negx):
        has_right_arm = True

        isct_pnt = LineString([arm_pnt_negx, arm_pnt_negx + (arm_pnt_negx-arm_pnt_posx)]).intersection(contour_str)
        isct_pnt = np.array(isct_pnt.coords[:])

        half_arm_len = norm(isct_pnt-arm_pnt_negx)
        x_0 = arm_pnt_negx[0] + 0.5*half_arm_len
        x_1 = arm_pnt_negx[0] + 2.0*half_arm_len

        chains = extract_contour_chain(contour, x_0, x_1)
        if len(chains) < 2:
            fixed_right = False
        else:
            fixed_right = True
            chain_lens = -np.array([len(chains[i]) for i in range(len(chains))])
            chain_idxs = np.argsort(chain_lens)[:2]
            chains = [chains[idx] for idx in chain_idxs]

        if fixed_right:
            chain_str_0 = MultiPoint([contour[i,:] for i in chains[0]])
            chain_str_1 = MultiPoint([contour[i,:] for i in chains[1]])
            points = ops.nearest_points(chain_str_0, chain_str_1)

            pair_idx0 = closest_point_idx(contour, np.array(points[0].coords[:]))
            pair_idx1 = closest_point_idx(contour, np.array(points[1].coords[:]))

            if contour[pair_idx0, 1] > contour[pair_idx1, 1]:
                pair_idx0, pair_idx1 = pair_idx1, pair_idx0

            shift = 1
            pair_idx0 -= shift
            pair_idx1 += shift

            if debug_path is not None:
                plt.plot(contour[pair_idx0, 0], contour[pair_idx0, 1], '+r', ms=15)
                plt.plot(contour[pair_idx1, 0], contour[pair_idx1, 1], '+r', ms=15)

                for i in chains[0]:
                    plt.plot(contour[i, 0], contour[i, 1],'+r', ms=5)

                for i in chains[1]:
                    plt.plot(contour[i, 0], contour[i, 1], '+b', ms=5)

            contour = np.concatenate([contour[:pair_idx0, :], contour[pair_idx1:, :]], axis = 0)

    if arm_inside_contour(contour_str, arm_pnt_posx):
        has_left_arm = True

        isct_pnt = LineString([arm_pnt_posx, arm_pnt_posx + (arm_pnt_posx-arm_pnt_negx)]).intersection(contour_str)
        isct_pnt = np.array(isct_pnt.coords[:])
        half_arm_len = norm(isct_pnt-arm_pnt_posx)

        x_0 = arm_pnt_posx[0] - 2.0*half_arm_len
        x_1 = arm_pnt_posx[0] - 0.5*half_arm_len
        chains = extract_contour_chain(contour, x_0, x_1)
        if len(chains) < 2:
            fixed_left = False
        else:
            chain_lens = -np.array([len(chains[i]) for i in range(len(chains))])
            chain_idxs = np.argsort(chain_lens)[:2]
            chains = [chains[idx] for idx in chain_idxs]
            fixed_left = True

        if fixed_left:
            chain_str_0 = MultiPoint([contour[i,:] for i in chains[0]])
            chain_str_1 = MultiPoint([contour[i,:] for i in chains[1]])
            points = ops.nearest_points(chain_str_0, chain_str_1)

            pair_idx0 = closest_point_idx(contour, np.array(points[0].coords[:]))
            pair_idx1 = closest_point_idx(contour, np.array(points[1].coords[:]))

            shift = 1
            if contour[pair_idx0, 1] > contour[pair_idx1, 1]:
                pair_idx0, pair_idx1 = pair_idx1, pair_idx0

            pair_idx0 += shift
            pair_idx1 -= shift

            if debug_path is not None:
                plt.plot(contour[pair_idx0, 0], contour[pair_idx0, 1], '+r', ms=15)
                plt.plot(contour[pair_idx1, 0], contour[pair_idx1, 1], '+r', ms=15)

                for i in chains[0]:
                    plt.plot(contour[i, 0], contour[i, 1],'+r', ms=5)

                for i in chains[1]:
                    plt.plot(contour[i, 0], contour[i, 1], '+b', ms=5)

            contour = np.concatenate([contour[:pair_idx1, :], contour[pair_idx0:, :]], axis = 0)

    if fixed_left or fixed_right:
        X, Y = util.smooth_contour(contour[:, 0], contour[:, 1], sigma=1)
        X, Y = util.resample_contour(X, Y, 150)
        contour = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

    if debug_path is not None:
        plt.plot(contour[:,0], contour[:,1], '-r')
        if fixed_left or fixed_right:
            plt.savefig(debug_path)

    return contour, has_left_arm, has_right_arm, fixed_left, fixed_right

#contour should be already pre-processed
def fix_bust_height(bust_contour, sup_points, ld_points, armscye_contour, debug_path = None):
    bust_contour    = preprocess_contour(bust_contour)
    armscye_contour = preprocess_contour(armscye_contour)

    arm_pnt_negx = np.array(sup_points['Bust_Arm_NegX'][:2])
    arm_pnt_posx = np.array(sup_points['Bust_Arm_PosX'][:2])

    n_bust_contour = bust_contour.shape[0]

    mid_bust = ld_points[14][:2]
    neg_bust = ld_points[12][:2]
    pos_bust = ld_points[13][:2]

    if debug_path is not None:
        plt.clf()
        plt.axes().set_aspect(1)

    #align the input contours by rotating its arm-arm axis to match to x axis
    hor_ax  = util.normalize(arm_pnt_posx - arm_pnt_negx)
    angle = np.arccos(np.dot(hor_ax, np.array([1.0, 0.0])))
    if hor_ax[1] > 0.0:
        angle = -angle
    cos_a   = np.cos(angle)
    sin_a   = np.sin(angle)

    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_center = mid_bust
    bust_contour = np.dot(rot_mat, (bust_contour - rot_center).T).T + rot_center

    armscye_contour = np.dot(rot_mat, (armscye_contour - rot_center).T).T + rot_center
    neg_bust = (np.dot(rot_mat, (neg_bust-rot_center).reshape(2,1)).T + rot_center).flatten()
    pos_bust = (np.dot(rot_mat, (pos_bust-rot_center).reshape(2,1)).T + rot_center).flatten()

    arm_pnt_negx = (np.dot(rot_mat, (arm_pnt_negx-rot_center).reshape(2,1)).T + rot_center).flatten()
    arm_pnt_posx = (np.dot(rot_mat, (arm_pnt_posx-rot_center).reshape(2,1)).T + rot_center).flatten()

    if debug_path is not None:
        plt.plot(armscye_contour[:,0], armscye_contour[:,1], '-c')
        plt.plot([arm_pnt_negx[0], arm_pnt_posx[0]], [arm_pnt_negx[1], arm_pnt_posx[1]], '-r')
        plt.plot(arm_pnt_negx[0], arm_pnt_negx[1], '+r')
        plt.plot(arm_pnt_posx[0], arm_pnt_posx[1], '+r')
        pass

    hor_ax = np.array([1.0, 0.0])
    ver_ax = np.array([0.0, 1.0])

    #assume that the bust base line is equavalent to the armscye contour (a bit higher than the bust contour)
    #we can't take the under bust contour as the bust base line because some people have big belly, which means their udner bust contour is even larger than their bust contour.
    armscye_contour_str = LineString([armscye_contour[i,:] for i in range(armscye_contour.shape[0])])
    bust_base_target = LineString([mid_bust-10*ver_ax, mid_bust + 10*ver_ax]).intersection(armscye_contour_str)
    bust_base_target = util.closest_point_points(mid_bust, bust_base_target)

    left_side_ids = []
    for idx in range(0, n_bust_contour):
        p = bust_contour[idx, :]
        if p[1] < bust_base_target[1] and p[0] <= bust_base_target[0]:
            left_side_ids.append(idx)
            #plt.plot(p[0], p[1], '+r', ms=5)

    right_side_ids = []
    for idx in range(0, n_bust_contour):
        p = bust_contour[idx, :]
        if p[1] < bust_base_target[1] and p[0] >= bust_base_target[0]:
            right_side_ids.append(idx)
            #plt.plot(p[0], p[1], '+r', ms=5)

    if len(left_side_ids) > 0 and len(right_side_ids) > 0:
        left_tip_idx  = np.argmin(bust_contour[left_side_ids, 1])
        right_tip_idx = np.argmin(bust_contour[right_side_ids, 1])
        left_tip_idx = left_side_ids[left_tip_idx]
        right_tip_idx = right_side_ids[right_tip_idx]

        left_side_ids_1  = [idx for idx in left_side_ids if bust_contour[idx, 0] <=  bust_contour[left_tip_idx, 0]]
        right_side_ids_1 = [idx for idx in right_side_ids if bust_contour[idx, 0] >= bust_contour[right_tip_idx, 0]]

        take_left = False
        tip_idx = right_tip_idx
        side_point_idxs = right_side_ids_1
        if bust_contour[left_tip_idx, 1] < bust_contour[right_tip_idx, 1]:
            tip_idx = left_tip_idx
            side_point_idxs = left_side_ids_1
            take_left = True

        side_bust_tip_point = bust_contour[tip_idx, :]

        side_mirror_points = []
        for idx in side_point_idxs:
            p = bust_contour[idx, :]
            p_mirror = util.mirror_point_through_axis(side_bust_tip_point, ver_ax, p)
            side_mirror_points.append(p_mirror)

        #scale mirrored points
        side_mirror_points = np.array(side_mirror_points)
        #plt.plot(side_mirror_points[:,0], side_mirror_points[:,1], '+r')
        x_range = np.max(side_mirror_points[:,0]) - np.min(side_mirror_points[:,0])
        target_range =  abs(side_bust_tip_point[0]- bust_base_target[0])
        x_scale =  target_range/x_range
        side_mirror_points[:, 0] = (side_mirror_points[:, 0] - side_bust_tip_point[0]) * x_scale  + side_bust_tip_point[0]
        #plt.plot(side_mirror_points[:,0], side_mirror_points[:,1], '+b')

        tmp_idxs = []
        for idx in range(n_bust_contour):
            if bust_contour[idx,0] >= bust_base_target[0] and bust_contour[idx,1] >0:
                tmp_idxs.append(idx)

        start_idx = np.argmin(bust_contour[tmp_idxs,0])
        start_idx = tmp_idxs[start_idx]
        bust_contour = roll_contour(bust_contour, -start_idx)
        mirror_centroid = bust_base_target
        if take_left:
            half_contour = bust_contour[left_tip_idx-start_idx:n_bust_contour, :]
            half_contour = np.concatenatgite([side_mirror_points[::-1, :], half_contour])
            half_contour = half_contour - mirror_centroid

            #mirror
            other_half_contour = copy(half_contour)
            other_half_contour[:,0] = -other_half_contour[:,0]

            final_contour = np.concatenate([other_half_contour[::-1,:], half_contour])
        else:
            half_contour = bust_contour[0:right_tip_idx-start_idx, :]
            half_contour = np.concatenate([half_contour, side_mirror_points[::-1, :]])
            half_contour = half_contour - mirror_centroid

            other_half_contour = copy(half_contour)
            other_half_contour[:,0] = -other_half_contour[:,0]

            final_contour = np.concatenate([half_contour, other_half_contour[::-1,:]])

        final_contour[:,0], final_contour[:,1] = util.smooth_contour(final_contour[:,0], final_contour[:,1], sigma=2)
        final_contour += mirror_centroid

        if debug_path is not None:
            plt.plot(mirror_centroid[0], mirror_centroid[1], '+b', ms=20)
            plt.plot(side_bust_tip_point[0], side_bust_tip_point[1], '+b', ms=20)
            plt.plot(final_contour[:, 0], final_contour[:, 1], '-r', label='fixed bust')
            plt.plot(bust_base_target[0], bust_base_target[1], '+r', ms=20)
            plt.plot(bust_contour[:, 0], bust_contour[:, 1], '-b', label='input bust')
            plt.legend(loc='upper right')
            plt.savefig(debug_path)

        return final_contour, True

    else:
        if debug_path is not None:
            plt.plot(bust_base_target[0], bust_base_target[1], '+r', ms=20)
            plt.plot(bust_contour[:, 0], bust_contour[:, 1], '-b', label='input bust')
            plt.legend(loc='upper right')
            plt.savefig(debug_path)

        return bust_contour, False

#contour: clockwise order
#landmarks: contain points inside armt parts
def remove_arm_from_bust_slice(contour, sup_points, debug_path = None):
    contour = preprocess_contour(contour)

    arm_pnt_negx = np.array(sup_points['Bust_Arm_NegX'][:2])
    arm_pnt_posx = np.array(sup_points['Bust_Arm_PosX'][:2])

    if debug_path is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(contour[:,0], contour[:,1], '-b')
        plt.plot(arm_pnt_negx[0], arm_pnt_negx[1], '+r')
        plt.plot(arm_pnt_posx[0], arm_pnt_posx[1], '+r')

    fixed_left = False
    fixed_right = False
    has_left_arm = False
    has_right_arm = False

    contour_str = LinearRing([contour[i,:] for i in range(contour.shape[0])])
    if arm_inside_contour(contour_str, arm_pnt_negx):
        has_right_arm = True

        isct_pnt = LineString([arm_pnt_negx, arm_pnt_negx + (arm_pnt_negx-arm_pnt_posx)]).intersection(contour_str)
        isct_pnt = np.array(isct_pnt.coords[:])

        half_arm_len = norm(isct_pnt-arm_pnt_negx)
        x_0 = arm_pnt_negx[0] + 1.1*half_arm_len
        x_1 = arm_pnt_negx[0] + 2.0*half_arm_len

        chains = extract_contour_chain(contour, x_0, x_1)
        if len(chains) < 2:
            fixed_right = False
        else:
            fixed_right = True
            chain_lens = -np.array([len(chains[i]) for i in range(len(chains))])
            chain_idxs = np.argsort(chain_lens)[:2]
            chains = [chains[idx] for idx in chain_idxs]

        if fixed_right:
            chain_str_0 = MultiPoint([contour[i,:] for i in chains[0]])
            chain_str_1 = MultiPoint([contour[i,:] for i in chains[1]])
            points = ops.nearest_points(chain_str_0, chain_str_1)

            pair_idx0 = closest_point_idx(contour, np.array(points[0].coords[:]))
            pair_idx1 = closest_point_idx(contour, np.array(points[1].coords[:]))

            if contour[pair_idx0, 1] > contour[pair_idx1, 1]:
                pair_idx0, pair_idx1 = pair_idx1, pair_idx0

            shift = 1
            pair_idx0 -= shift
            pair_idx1 += shift

            if debug_path is not None:
                plt.plot(contour[pair_idx0, 0], contour[pair_idx0, 1], '+r', ms=15)
                plt.plot(contour[pair_idx1, 0], contour[pair_idx1, 1], '+r', ms=15)

                for i in chains[0]:
                    plt.plot(contour[i, 0], contour[i, 1],'+r', ms=5)

                for i in chains[1]:
                    plt.plot(contour[i, 0], contour[i, 1], '+b', ms=5)

            contour = np.concatenate([contour[:pair_idx0, :], contour[pair_idx1:, :]], axis = 0)

    if arm_inside_contour(contour_str, arm_pnt_posx):
        has_left_arm = True

        isct_pnt = LineString([arm_pnt_posx, arm_pnt_posx + (arm_pnt_posx-arm_pnt_negx)]).intersection(contour_str)
        isct_pnt = np.array(isct_pnt.coords[:])
        half_arm_len = norm(isct_pnt-arm_pnt_posx)

        x_0 = arm_pnt_posx[0] - 2.0*half_arm_len
        x_1 = arm_pnt_posx[0] - 1.1*half_arm_len
        chains = extract_contour_chain(contour, x_0, x_1)
        if len(chains) < 2:
            fixed_left = False
        else:
            chain_lens = -np.array([len(chains[i]) for i in range(len(chains))])
            chain_idxs = np.argsort(chain_lens)[:2]
            chains = [chains[idx] for idx in chain_idxs]
            fixed_left = True

        if fixed_left:
            chain_str_0 = MultiPoint([contour[i,:] for i in chains[0]])
            chain_str_1 = MultiPoint([contour[i,:] for i in chains[1]])
            points = ops.nearest_points(chain_str_0, chain_str_1)

            pair_idx0 = closest_point_idx(contour, np.array(points[0].coords[:]))
            pair_idx1 = closest_point_idx(contour, np.array(points[1].coords[:]))

            shift = 1
            if contour[pair_idx0, 1] > contour[pair_idx1, 1]:
                pair_idx0, pair_idx1 = pair_idx1, pair_idx0

            pair_idx0 += shift
            pair_idx1 -= shift

            if debug_path is not None:
                plt.plot(contour[pair_idx0, 0], contour[pair_idx0, 1], '+r', ms=15)
                plt.plot(contour[pair_idx1, 0], contour[pair_idx1, 1], '+r', ms=15)

                for i in chains[0]:
                    plt.plot(contour[i, 0], contour[i, 1],'+r', ms=5)

                for i in chains[1]:
                    plt.plot(contour[i, 0], contour[i, 1], '+b', ms=5)

            contour = np.concatenate([contour[:pair_idx1, :], contour[pair_idx0:, :]], axis = 0)

    if fixed_left or fixed_right:
        X, Y = util.smooth_contour(contour[:, 0], contour[:, 1], sigma=1)
        X, Y = util.resample_contour(X, Y, 150)
        contour = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

    if debug_path is not None:
        plt.plot(contour[:,0], contour[:,1], '-r')
        if fixed_left or fixed_right:
            plt.savefig(debug_path)

    return contour, has_left_arm, has_right_arm, fixed_left, fixed_right

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="")
    ap.add_argument("-l", "--landmark", required=True, help="")
    ap.add_argument("-o", "--output", required=True, help="")
    args = vars(ap.parse_args())
    IN_DIR   = args['input']
    LD_DIR   = args['landmark']
    OUT_DIR  = args['output']

    DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/'

    DIR_NO_ARM_CUT_OFF = f'{DEBUG_DIR}/bust_arm_cut_off/'
    shutil.rmtree(DIR_NO_ARM_CUT_OFF)
    os.makedirs(DIR_NO_ARM_CUT_OFF)

    os.makedirs(OUT_DIR, exist_ok=True)

    all_paths = [path for path in Path(IN_DIR).glob('*.*')]
    n_error_contour = 0
    n_fixed_contour = 0
    for i, path in enumerate(all_paths):

        print(f'{i}, {path.stem}')

        with open(path, 'rb') as file:
            slc_contours = pickle.load(file)
            lens = np.array([len(contour) for contour in slc_contours])
            contour = slc_contours[np.argmax(lens)]
            contour = np.array(contour)
            contour = contour[:, :2]

        ld_path = f'{LD_DIR}/{path.stem}.pkl'
        assert os.path.exists(ld_path)

        with open(ld_path, 'rb') as file:
            landmarks = pickle.load(file)

        debug_path = f'{DIR_NO_ARM_CUT_OFF}{path.stem}.png'
        contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_bust_slice(contour, landmarks, debug_path=debug_path)
        if has_left or has_right:
            n_error_contour += 1
            if not (has_left != fixed_left) or not (has_right != fixed_right):
                n_fixed_contour += 1

        #with open(f'{OUT_DIR}/{path.stem}.pkl', 'wb') as out_file:
        #    pickle.dump(out_file, contour)

    print(f'n error contour = {n_error_contour}, n fixed contour = {n_fixed_contour}')
