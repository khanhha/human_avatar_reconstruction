import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pathlib import Path
import common.util as util
import common.util_math as util_math
from common.util import sample_contour_radial
from src.slice_preprocess.caesar_slc_fix_bust import \
    remove_arm_from_bust_slice, remove_arm_from_under_bust_slice, remove_arm_from_armscye_slice, fix_bust_height, preprocess_contour
import multiprocessing
import sys
import shutil

G_cur_file_path = Path()
IN_DIR = ''
OUT_DIR = ''
SUPPOINT_DIR = ''
LDPOINT_DIR = ''
DEBUG_DIR = ''

def plot_segment(p0, p1, type):
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], type)

def resample_contour(X, Y, n_point = 150, debug_path = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = 0.5 * (X[idx_ymin] + X[idx_ymax])

    X, Y = util.resample_contour(X, Y, n_point)

    #find the starting contour point
    min_dst = np.inf
    min_idx = 0
    i = 0
    for x, y in zip(X,Y):
        if x > center_x:
            y_dif = abs(y - center_y)
            if y_dif < min_dst:
                min_dst = y_dif
                min_idx = i
        i += 1

    X = np.roll(X, -min_idx)
    Y = np.roll(Y, -min_idx)
    #print('starting index: ', min_idx)

    #fix bad points
    # jump = np.sqrt(np.diff(X) ** 2 + np.diff(X) ** 2)
    # smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
    # limit = 2 * np.median(smooth_jump)  # factor 2 is arbitrary
    # xn, yn = X[:-1], Y[:-1]
    # X = xn[(jump > 0) & (smooth_jump < limit)]
    # Y = yn[(jump > 0) & (smooth_jump < limit)]

    #resample
    # tck, u = splprep([X, Y], s=0)
    # u_1 = np.linspace(0.0, 1.0, 150)
    # X, Y = splev(u_1, tck)

    if debug_path is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(X, Y, 'b-')
        plt.plot(center_x, center_y, 'r+')
        plt.plot(X[0], Y[0], 'r+', ms = 20)
        plt.plot(X[5], Y[5], 'g+', ms = 20)
        plt.savefig(debug_path)
    return X, Y

def load_contour(type, path):
    with open(path, 'rb') as file:
        slc_contours = pickle.load(file)
    assert len(slc_contours) != 0
    if util.is_leg_contour(type):
        lens = np.array([len(contour) for contour in slc_contours])
        cnt_0, cnt_1 = np.argsort(-lens)[:2]
        contour_0 = np.array(slc_contours[cnt_0])
        contour_1 = np.array(slc_contours[cnt_1])
        center_0  = np.mean(contour_0, axis=0)
        center_1  = np.mean(contour_1, axis=0)
        if center_1[0] > center_0[0]:
            return contour_1
        else:
            return contour_0
        pass
    else:
        lens = np.array([len(contour) for contour in slc_contours])
        contour = slc_contours[np.argmax(lens)]
        contour = np.array(contour)
        contour = contour[:, :2]
        return contour



def radial_code(points, D, half = True):
    n_points = points.shape[0]
    idx_half = int(n_points / 2) + 1
    if half:
        idx_max = int(n_points/2) + 1
    else:
        idx_max = n_points

    feature = []

    for i in range(1, idx_max):
        dy = points[i,1] - points[i-1,1]
        dx = points[i,0] - points[i-1,0]
        c = dy/dx
        #c = np.clip(c, a_min=-100.0, a_max=100.0)
        feature.append(c)

    D_mid =  norm(points[0] - points[idx_half])
    r1 = D_mid / D
    r2 = norm(points[0]) /  D_mid
    feature.extend([r1, r2])

    return feature

def torso_contour_center(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = 0.5 * (X[idx_ymin] + X[idx_ymax])
    center = np.array([center_x, center_y])

    return center

def torso_contour_w_d(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)

    W = Y[idx_ymax] - Y[idx_ymin]
    D = X[idx_xmax] - X[idx_xmin]

    return W, D

def convert_torso_contour_to_radial_code(X, Y, n_sample, path_out = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = 0.5 * (X[idx_ymin] + X[idx_ymax])
    center = np.array([center_x, center_y])

    idx_half = int(n_sample/2) + 1

    W = Y[idx_ymax] - Y[idx_ymin]
    D = X[idx_xmax] - X[idx_xmin]

    points = sample_contour_radial(X, Y, center, n_sample)

    if path_out is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(X, Y, 'b-')
        plt.plot(center_x, center_y, 'r+')

    feature = []
    for i in range(1, idx_half):
        dy = points[i][1] - points[i-1][1]
        dx = points[i][0] - points[i-1][0]
        c = dy/dx
        feature.append(c)

    D_mid =  norm(points[0] - points[idx_half])
    r1 = D_mid / D
    r2 = norm(points[0]) /  D_mid
    feature.extend([r1, r2])

    if path_out is not None:
        for p in points:
            p = center + p
            plt.plot([center_x, p[0]], [center_y, p[1]], 'r-')
            plt.plot(p[0], p[1], 'b+')

        points_1 = util.reconstruct_torso_slice_contour(feature, D, W)
        for i in range(points_1.shape[1]):
              p = points_1[:,i]
              p = center + p
              plt.plot(p[0], p[1], 'r+')
        plt.savefig(path_out)
        #plt.show()

    return np.array(feature), W, D

def leg_contour_w_d(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)

    W = Y[idx_ymax] - Y[idx_ymin]
    D = X[idx_xmax] - X[idx_xmin]

    return W, D

def convert_leg_contour_to_radial_code(X, Y, n_sample, path_out = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymax]
    center = np.array([center_x, center_y])

    W = Y[idx_ymax] - Y[idx_ymin]
    D = X[idx_xmax] - X[idx_xmin]

    if path_out is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(X, Y, 'b-')
        plt.plot(center_x, center_y, 'r+')
        plt.plot(X[idx_ymax], Y[idx_ymax], 'go', ms=5)
        plt.plot(X[idx_ymin], Y[idx_ymin], 'go', ms=5)
        #plt.show()

    points = sample_contour_radial(X, Y, center, n_sample)

    points = np.array(points)
    feature = radial_code(points, D, half=False)

    if path_out is not None:
        for p in points:
            p = center + p
            plt.plot([center_x, p[0]], [center_y, p[1]], 'r-')
            plt.plot(p[0], p[1], 'b+')

        points_1 = util.reconstruct_leg_slice_contour(feature, D, W)
        for i in range(points_1.shape[1]):
              p = points_1[:,i]
              p = center + p
              plt.plot(p[0], p[1], 'r+')
        plt.savefig(path_out)
        #plt.show()

    return np.array(feature), W, D


#estimate leg axis, running from right to left (neg x => pos x)
def leg_hor_axis(ld_points):
    right_0 = ld_points[20][:2]
    right_1 = ld_points[16][:2]
    left_0 = ld_points[22][:2]
    left_1 = ld_points[18][:2]

    hor_dir = 0.5*(left_0+left_1) - 0.5*(right_0+right_1)

    return util_math.normalize(hor_dir)

def align_leg_contour(X, Y, ld_points):
    hor_ax = leg_hor_axis(ld_points)

    angle = np.arccos(np.dot(hor_ax, np.array([1.0, 0.0])))
    if hor_ax[1] > 0.0:
        angle = -angle
    cos_a   = np.cos(angle)
    sin_a   = np.sin(angle)

    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_center = np.array([np.mean(X), np.mean(Y)])
    contour = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    contour_1 = np.dot(rot_mat, (contour - rot_center).T).T + rot_center

    #test_p0 = rot_center - 0.5*hor_ax
    #test_p1 = rot_center + 0.5*hor_ax
    #plt.clf()
    #plt.axes().set_aspect(1.0)
    #plt.plot([test_p0[0], test_p1[0]], [test_p0[1], test_p1[1]], '-r')
    #plt.plot(X, Y, '-b')
    #plt.plot(contour_1[:,0], contour_1[:,1], '-r')
    #plt.show()
    return contour_1[:,0], contour_1[:,1]

from copy import deepcopy
#make contour start from its cleavage point
#make contour symmetric
def fix_crotch_contour(X, Y):
    #top-right part
    tr_mask = np.bitwise_and(X > 0, Y > 0)
    tmp_X = deepcopy(X)
    tmp_X[~tr_mask] = -np.inf
    above_idx = np.argmax(tmp_X)

    #bottom right pat
    br_mask = np.bitwise_and(X > 0, Y < 0)
    tmp_X = deepcopy(X)
    tmp_X[~br_mask] = -np.inf
    below_idx = np.argmax((tmp_X))

    #find the cleavage
    #candidate mask
    cdd_mask = np.bitwise_and(Y > Y[below_idx], Y < Y[above_idx])
    cdd_mask = np.bitwise_and(cdd_mask, X > 0)
    tmp_X = deepcopy(X)
    tmp_X[~cdd_mask] = np.inf
    cleavage_idx = int( np.argmin(tmp_X))

    #rolling to make the cleaveage point as the first point of contour
    X = np.roll(X, -cleavage_idx)
    Y = np.roll(Y, -cleavage_idx)
    Y = Y - Y[0] #make the contour zero-center along Y direction
    #from now on cleavage_idx = 0

    mask =  X < 0
    tmp_Y = deepcopy(Y)
    tmp_Y[~mask] = np.inf
    diff_Y = np.abs(Y[0] - tmp_Y)
    half_idx = np.argmin(diff_Y)

    half_X = X[:half_idx+1]
    half_Y = Y[:half_idx+1]
    mask = half_Y >= 0.0
    half_X = half_X[mask]
    half_Y = half_Y[mask]

    X1 = np.concatenate([half_X,  half_X[1:][::-1]], axis=0)
    Y1 = np.concatenate([half_Y, -half_Y[1:][::-1]], axis=0)
    # print(np.diff(X1))
    # print(np.diff(Y1))
    # plt.clf()
    # plt.axes().set_aspect(1.0)
    # plt.plot(X1, Y1, '-b')
    # plt.plot(X1, Y1, '+r')
    # plt.show()

    return X1, Y1

def process_leg_contour(path, contour, sup_points, ld_points):
    contour = preprocess_contour(contour, resolution=150)

    Y = contour[:, 0]
    X = contour[:, 1]

    X, Y = align_leg_contour(X, Y, ld_points)

    X, Y = resample_contour(X, Y)

    W, D = leg_contour_w_d(X, Y)

    return X, Y, W, D

def process_torso_contour(path, contour, sup_points, ld_points):
    # torso contour
    align_anchor_pos_x = True
    if slc_id == 'Armscye':
        align_anchor_pos_x = False

        debug_armscye_path = f'{DEBUG_ARMSCYE_DIR}/{path.stem}.png'
        contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_armscye_slice(contour,
                                                                                           sup_points=sup_points,
                                                                                           debug_path=debug_armscye_path)
        if has_left != fixed_left or has_right != fixed_right:
            raise Exception('Failed armscye slice')

    elif slc_id == 'Bust':
        align_anchor_pos_x = False

        armscye_path = f'{IN_DIR}/Armscye/{path.name}'
        armscye_contour = load_contour('Armscye', armscye_path)

        debug_bust_path = f'{DEBUG_BUST_DIR}/{path.stem}.png'
        contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_bust_slice(contour,
                                                                                           sup_points=sup_points,
                                                                                           debug_path=debug_bust_path)
        if has_left != fixed_left or has_right != fixed_right:
            raise Exception('Failed bust slice')

        debug_bust_height_path = f'{DEBUG_BUST_HEIGHT_DIR}/{path.stem}.png'
        # contour, ok = fix_bust_height(contour, sup_points=sup_points, ld_points=ld_points,
        #                               armscye_contour=armscye_contour, debug_path=debug_bust_height_path)

    elif slc_id == 'Aux_UnderBust_Bust_0':
        align_anchor_pos_x = False

        armscye_path = f'{IN_DIR}/Armscye/{path.name}'
        armscye_contour = load_contour('Armscye', armscye_path)

        arm_pnt_negx = np.array(sup_points['Aux_UnderBust_Bust_0_NegX'][:2])
        arm_pnt_posx = np.array(sup_points['Aux_UnderBust_Bust_0_PosX'][:2])
        debug_bust_path = f'{DEBUG_UNDERBUST_BUST_DIR}/{path.stem}_bust.png'
        contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_under_bust_slice(contour,
                                                                                                 arm_pnt_negx=arm_pnt_negx,
                                                                                                 arm_pnt_posx=arm_pnt_posx,
                                                                                                 debug_path=debug_bust_path)
        if has_left != fixed_left or has_right != fixed_right:
            raise Exception('Failed under_bust slice')

        debug_underbust_height_path = f'{DEBUG_UNDERBUST_HEIGHT_DIR}/{path.stem}.png'
        #contour, ok = fix_bust_height(contour, sup_points=sup_points, ld_points=ld_points,
        #                              armscye_contour=armscye_contour, debug_path=debug_underbust_height_path)
    else:
        contour = preprocess_contour(contour, resolution=150)
        pass

    # transpose, swap X and Y to make the coordinate system more natural to the contour shape
    Y = contour[:, 0]
    X = contour[:, 1]

    X, Y = util.align_torso_contour(X, Y, anchor_pos_x=align_anchor_pos_x, debug_path=None)


    if X is None or Y is None:
        raise Exception('failed torso contour alignment')
    if slc_id == 'Crotch' or 'Aux_Crotch_Hip_' in slc_id:
        X, Y = fix_crotch_contour(X, Y)

    X, Y = resample_contour(X, Y)

    debug_align_path = f'{DEBUG_ALIGN_DIR}/{path.stem}.png'
    X, Y = util.symmetrize_contour(X, Y, debug_path=debug_align_path)

    #X, Y = util.smooth_contour(X, Y, sigma=2.0)

    W, D = torso_contour_w_d(X, Y)

    return X, Y, W, D


import warnings
def run_process_slice_contours(process_id, slc_id, paths, shared_data):
    n_paths = len(paths)
    #print(f'process {self.id} started. n_paths = {n_paths}')

    Ws = []
    Ds = []
    Cs = []
    Ns = []

    warnings.filterwarnings('error')

    for i, path in enumerate(paths):
        if i % 20 == 0:
            print(f'process {process_id}: {100.0 * float(i) / float(n_paths)}%')

        suppoints_path = f'{SUPPOINT_DIR}/{path.stem}.pkl'
        assert os.path.exists(suppoints_path)
        with open(suppoints_path, 'rb') as file:
            sup_points = pickle.load(file)

        ld_path = f'{LDPOINT_DIR}/{path.stem}.pkl'
        assert os.path.exists(ld_path)
        with open(ld_path, 'rb') as file:
            ld_points = pickle.load(file)

        contour = load_contour(slc_id, path)

        # crotch_pos = ld_points[72]
        # plt.clf()
        # plt.axes().set_aspect(1.0)
        # plt.plot(contour[:,0], contour[:,1])
        # plt.plot(crotch_pos[0], crotch_pos[1], '+r')
        # plt.show()

        if util.is_leg_contour(slc_id):
            try:
                X, Y, W, D = process_leg_contour(path, contour, sup_points=sup_points, ld_points=ld_points)
            except Exception as exp:
                print(path.stem, exp, file=sys.stderr)
                continue
        else:
            try:
                X, Y, W, D = process_torso_contour(path, contour, sup_points=sup_points, ld_points=ld_points)
            except Exception as exp:
                print(path.stem, f'khanh exp: {exp}', file=sys.stderr)
                continue
            except RuntimeWarning as warn:
                print(path.stem, f'khanh warn: {warn}', file=sys.stderr)
                continue

        #acculumate one more slice record
        Ws.append(W)
        Ds.append(D)
        Cs.append(np.vstack([X,Y]))
        Ns.append(path.stem)

    shared_data[process_id] = {'Ws':Ws, 'Ds':Ds, 'Cs':Cs, 'Ns':Ns}

    print(f'process {process_id} finished. len(Ws)={len(Ws)}')

if __name__  == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True, help="")
    ap.add_argument("-o", "--output", type=str, required=True, help="")
    ap.add_argument("-p", "--suppoint", type=str, required=True, help="")
    ap.add_argument("-l", "--ldpoint", type=str, required=True, help="")
    ap.add_argument("-ids", "--slc_ids", type=str,  required=True, help="")
    ap.add_argument("-np", "--nprocess", type=int,  required=False, default=1 ,help="")

    args = ap.parse_args()
    IN_DIR  = args.input
    OUT_DIR = args.output
    SUPPOINT_DIR   = args.suppoint
    LDPOINT_DIR   = args.ldpoint
    slc_ids = args.slc_ids
    n_processes = args.nprocess

    DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/debug/'
    os.makedirs(DEBUG_DIR, exist_ok=True)

    all_slc_ids = [path.stem for path in Path(IN_DIR).glob('./*')]
    if slc_ids == 'all':
        slc_ids = all_slc_ids
    elif slc_ids == 'torso':
        slc_ids = ['Crotch', 'Aux_Crotch_Hip_0', 'Aux_Crotch_Hip_1', 'Aux_Crotch_Hip_1', 'Aux_Crotch_Hip_2', 'Hip'] + \
                        ['Aux_Hip_Waist_0', 'Aux_Hip_Waist_1', 'Waist'] + \
                        ['Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1', 'Aux_Waist_UnderBust_2', 'UnderBust', 'Bust']
    else:
        slc_ids = slc_ids.split(',')
        for id in slc_ids:
            assert id in all_slc_ids, f'{id}: unrecognized slice id'

    failed_slice_paths = []
    for slc_id in slc_ids:
        SLICE_DIR = f'{IN_DIR}/{slc_id}/'
        print(f'start processing slice {slc_id}')

        DEBUG_ALIGN_DIR = f'{DEBUG_DIR}/{slc_id}_align/'
        shutil.rmtree(DEBUG_ALIGN_DIR, ignore_errors=True)
        os.makedirs(DEBUG_ALIGN_DIR, exist_ok=True)

        DEBUG_RADIAL_DIR = f'{DEBUG_DIR}/{slc_id}_radial/'
        shutil.rmtree(DEBUG_RADIAL_DIR, ignore_errors=True)
        os.makedirs(DEBUG_RADIAL_DIR, exist_ok=True)

        if slc_id == 'Armscye':
            DEBUG_ARMSCYE_DIR = f'{DEBUG_DIR}/{slc_id}_armscye_cutoff/'
            shutil.rmtree(DEBUG_ARMSCYE_DIR, ignore_errors=True)
            os.makedirs(DEBUG_ARMSCYE_DIR, exist_ok=True)

        if slc_id == 'Bust':
            DEBUG_BUST_DIR = f'{DEBUG_DIR}/{slc_id}_bust_cutoff/'
            shutil.rmtree(DEBUG_BUST_DIR, ignore_errors=True)
            os.makedirs(DEBUG_BUST_DIR, exist_ok=True)

            DEBUG_BUST_HEIGHT_DIR = f'{DEBUG_DIR}/{slc_id}_bust_height/'
            shutil.rmtree(DEBUG_BUST_HEIGHT_DIR, ignore_errors=True)
            os.makedirs(DEBUG_BUST_HEIGHT_DIR, exist_ok=True)

        if slc_id == 'Aux_UnderBust_Bust_0':
            DEBUG_UNDERBUST_BUST_DIR = f'{DEBUG_DIR}/{slc_id}_aux_underbust_bust_0_cutoff/'
            shutil.rmtree(DEBUG_UNDERBUST_BUST_DIR, ignore_errors=True)
            os.makedirs(DEBUG_UNDERBUST_BUST_DIR, exist_ok=True)

            DEBUG_UNDERBUST_HEIGHT_DIR = f'{DEBUG_DIR}/{slc_id}_underbust_height/'
            shutil.rmtree(DEBUG_UNDERBUST_HEIGHT_DIR, ignore_errors=True)
            os.makedirs(DEBUG_UNDERBUST_HEIGHT_DIR, exist_ok=True)

        slc_paths = [path for path in Path(SLICE_DIR).glob('*.pkl')]
        #slc_paths = slc_paths[:3]
        n_paths = len(slc_paths)
        processes = []
        npath_per_process = int(n_paths / n_processes)

        manager = multiprocessing.Manager()
        shared_data = manager.dict()

        for i in range(n_processes):
            sub_path_list = slc_paths[i*npath_per_process:(i + 1) * npath_per_process]
            process = multiprocessing.Process(target=run_process_slice_contours, args=(i, slc_id, sub_path_list, shared_data))
            processes.append(process)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        Ws, Ds, Cs, Ns = [], [], [], []
        print('syncronizing data across processes')
        for id, data in shared_data.items():
            n_Ws = len(data['Ws'])
            print(f'{id}, len(Ws)={n_Ws}')
            assert len(data['Ws']) > 0
            assert len(data['Ds']) > 0
            Ws.extend(data['Ws'])
            Ds.extend(data['Ds'])
            Cs.extend(data['Cs'])
            Ns.extend(data['Ns'])

        #dump all records of that slice
        n_contour = len(Ns)

        feature_dir_out = f'{OUT_DIR}/{slc_id}/'
        os.makedirs(feature_dir_out, exist_ok=True)

        for i in range(n_contour):
            W, D, C = Ws[i], Ds[i], Cs[i]
            name = Ns[i]
            with open(f'{feature_dir_out}{name}.pkl', 'wb') as file:
                 pickle.dump({'W':W, 'D':D, 'cnt':C}, file)

    print('failed slice paths')
    print(failed_slice_paths)
    print(f'n failed slices = {len(failed_slice_paths)}')