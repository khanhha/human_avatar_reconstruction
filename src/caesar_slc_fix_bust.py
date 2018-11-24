import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import os
import sys
from shapely.geometry import LinearRing, LineString, Point, MultiPoint
from shapely.ops import  nearest_points
import shutil
import src.util as util
from numpy.linalg import norm

def preprocess_contour(contour):
    X, Y = util.smooth_contour(contour[:, 0], contour[:, 1], sigma=2)
    X, Y = util.resample_contour(X, Y, 200)

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

def inside_contour(contour, p):
    if p[0] > 0.0:
        p1 = p + 2.0*np.array([1, 0])
    else:
        p1 = p + 2.0*np.array([-1, 0])

    contour_str = LinearRing([contour[i,:] for i in range(contour.shape[0])])
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
def remove_arm_from_bust_slice(contour, landmarks, debug_path = None):

    contour = preprocess_contour(contour)

    if debug_path is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(contour[:,0], contour[:,1], '-b')

    ld_negx = np.array(landmarks['Bust_Arm_NegX'][:2])
    ld_posx = np.array(landmarks['Bust_Arm_PosX'][:2])
    hor_dir = ld_posx - ld_negx

    fixed_left = False
    fixed_right = False
    has_left_arm = False
    has_right_arm = False

    if inside_contour(contour, ld_negx):
        has_right_arm = True
        x_0 = ld_negx[0] + 0.01*norm(hor_dir)
        x_1 = ld_negx[0] + 0.15*norm(hor_dir)

        chains = extract_contour_chain(contour, x_0, x_1)
        if len(chains) == 2:
            fixed_right = True

        if fixed_right:
            chain_str_0 = MultiPoint([contour[i,:] for i in chains[0]])
            chain_str_1 = MultiPoint([contour[i,:] for i in chains[1]])
            points = nearest_points(chain_str_0, chain_str_1)

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

    if inside_contour(contour, ld_posx):
        has_left_arm = True
        x_0 = ld_posx[0] - 0.15*norm(hor_dir)
        x_1 = ld_posx[0] - 0.01*norm(hor_dir)
        chains = extract_contour_chain(contour, x_0, x_1)
        if len(chains) == 2:
            fixed_left = True

        if fixed_left:

            chain_str_0 = MultiPoint([contour[i,:] for i in chains[0]])
            chain_str_1 = MultiPoint([contour[i,:] for i in chains[1]])
            points = nearest_points(chain_str_0, chain_str_1)

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
