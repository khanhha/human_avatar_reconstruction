import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing, LineString, Point, Polygon
import shapely.affinity as affinity
from numpy.linalg import norm
from pathlib import Path
import math
import src.util as util
from src.caesar_slc_fix_bust import remove_arm_from_bust_slice, remove_arm_from_under_bust_slice, fix_bust_height
from scipy.spatial import ConvexHull
import scipy.ndimage as ndimage
import shutil
from scipy.interpolate import splev, splrep, splprep, splev

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
        plt.plot(X, Y, 'b+')
        plt.plot(center_x, center_y, 'r+')
        plt.plot(X[0], Y[0], 'r+', ms = 20)
        plt.show()
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

def sample_contour_radial(X, Y, center, n_sample):
    contour = LinearRing([(x,y) for x, y in zip(X,Y)])
    extend_dst =  (np.max(X)-np.min(X))
    angle_step = (2.0*np.pi)/float(n_sample)
    points = []
    for i in range(n_sample):
        x = np.cos(i*angle_step)
        y = np.sin(i*angle_step)
        p = center + extend_dst * np.array([x,y])
        isect_ret = LineString([(center[0], center[1]), (p[0],p[1])]).intersection(contour)
        if isect_ret.geom_type == 'Point':
            isect_p = np.array(isect_ret.coords[:]).flatten()
        elif isect_ret.geom_type == 'MultiPoint':
            isect_p = np.array(isect_ret[0].coords[:]).flatten()
        else:
            #assert False, 'unsupported intersection type'
            return points, False
        isect_p = isect_p - center
        points.append(isect_p)

    return points, True

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
        if np.isinf(c):
            print(f'zero division dy/dx: {dy}/{dx}')
        #c = np.clip(c, a_min=-100.0, a_max=100.0)
        feature.append(c)

    D_mid =  norm(points[0] - points[idx_half])
    r1 = D_mid / D
    r2 = norm(points[0]) /  D_mid
    feature.extend([r1, r2])

    return feature

def convert_torso_contour_to_radial_code(X, Y, n_sample, path_out = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = 0.5 * (X[idx_ymin] + X[idx_ymax])
    center = np.array([center_x, center_y])

    idx_half = int(n_sample/2) + 1

    W = Y[idx_ymax] - Y[idx_ymin]
    D = X[idx_xmax] - X[idx_xmin]

    if path_out is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(X, Y, 'b-')
        plt.plot(center_x, center_y, 'r+')

    points, ok = sample_contour_radial(X, Y, center, n_sample)

    feature = []
    for i in range(1, idx_half):
        dy = points[i][1] - points[i-1][1]
        dx = points[i][0] - points[i-1][0]
        c = dy/dx
        if np.isinf(c):
            print(f'zero division dy/dx: {dy}/{dx}')
        #c = np.clip(c, a_min=-100.0, a_max=100.0)
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
        plt.show()

    return np.array(feature), W, D

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

    points, ok = sample_contour_radial(X, Y, center, n_sample)
    if ok is False:
        raise Exception('failed to calc radial points')

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

    return util.normalize(hor_dir)

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

def process_torso_contour(contour, sup_points, ld_points):
    pass

def process_leg_contour(path, contour, sup_points, ld_points):
    Y = contour[:, 0]
    X = contour[:, 1]
    X, Y = align_leg_contour(X, Y, ld_points)
    X, Y = resample_contour(X, Y)
    X, Y = util.smooth_contour(X, Y, sigma=2.0)

    debug_path_out = f'{DEBUG_RADIAL_DIR}/{path.stem}.png'
    feature, W, D = convert_leg_contour_to_radial_code(X, Y, 8, path_out=debug_path_out)
    return X, Y, W, D, feature

import multiprocessing
def run_process_slice_contours(process_id, slc_id, paths, shared_data):
    n_paths = len(paths)
    #print(f'process {self.id} started. n_paths = {n_paths}')

    Ws = []
    Ds = []
    Fs = []
    Cs = []
    Ns = []

    for i, path in enumerate(paths):
        if i % 20 == 0:
            print(f'process {process_id}: {100.0 * float(i) / float(n_paths)}%')

        #print(i, str(path))

        #if 'csr4205a' not in path.name:
        #    continue

        suppoints_path = f'{SUPPOINT_DIR}/{path.stem}.pkl'
        assert os.path.exists(suppoints_path)
        with open(suppoints_path, 'rb') as file:
            sup_points = pickle.load(file)

        ld_path = f'{LDPOINT_DIR}/{path.stem}.pkl'
        assert os.path.exists(ld_path)
        with open(ld_path, 'rb') as file:
            ld_points = pickle.load(file)

        contour = load_contour(slc_id, path)

        if util.is_leg_contour(slc_id):
            try:
                X, Y, W, D, feature = process_leg_contour(path, contour, sup_points=sup_points, ld_points=ld_points)
            except Exception as exp:
                print(path.stem, exp)
                continue
        else:
            #torso contour
            align_anchor_pos_x = True
            if slc_id == 'Bust':
                align_anchor_pos_x = False

                armscye_path = f'{IN_DIR}/Armscye/{path.name}'
                armscye_contour = load_contour('Armscye', armscye_path)

                debug_bust_path = f'{DEBUG_BUST_DIR}/{path.stem}.png'
                contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_bust_slice(contour, sup_points=sup_points, debug_path=debug_bust_path)
                if has_left != fixed_left or has_right != fixed_right:
                    failed_slice_paths.append(path)

                debug_bust_height_path = f'{DEBUG_BUST_HEIGHT_DIR}/{path.stem}.png'
                contour, ok = fix_bust_height(contour, sup_points=sup_points, ld_points=ld_points, armscye_contour=armscye_contour, debug_path=debug_bust_height_path)

            if slc_id == 'Aux_UnderBust_Bust_0':
                align_anchor_pos_x = False

                armscye_path = f'{IN_DIR}/Armscye/{path.name}'
                armscye_contour = load_contour('Armscye', armscye_path)

                arm_pnt_negx = np.array(sup_points['Aux_UnderBust_Bust_0_NegX'][:2])
                arm_pnt_posx = np.array(sup_points['Aux_UnderBust_Bust_0_PosX'][:2])
                debug_bust_path = f'{DEBUG_UNDERBUST_BUST_DIR}/{path.stem}_bust.png'
                contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_under_bust_slice(contour, arm_pnt_negx=arm_pnt_negx, arm_pnt_posx=arm_pnt_posx, debug_path=debug_bust_path)
                if has_left != fixed_left or has_right != fixed_right:
                    failed_slice_paths.append(path)

                debug_underbust_height_path = f'{DEBUG_UNDERBUST_HEIGHT_DIR}/{path.stem}.png'
                contour, ok = fix_bust_height(contour, sup_points=sup_points, ld_points=ld_points, armscye_contour=armscye_contour, debug_path=debug_underbust_height_path)

            #transpose, swap X and Y to make the coordinate system more natural to the contour shape
            Y = contour[:, 0]
            X = contour[:, 1]

            debug_align_path = f'{DEBUG_ALIGN_DIR}/{path.stem}.png'
            X, Y = util.align_torso_contour(X, Y, anchor_pos_x= align_anchor_pos_x, debug_path=debug_align_path)
            if X is None or Y is None:
                failed_slice_paths.append(path)
                continue

            X, Y = resample_contour(X, Y)
            X, Y = util.smooth_contour(X,Y, sigma=2.0)

            debug_path_out = f'{DEBUG_RADIAL_DIR}/{path.stem}.png'
            feature, W, D = convert_torso_contour_to_radial_code(X, Y, 16, path_out=debug_path_out)

        #acculumate one more slice record
        Ws.append(W)
        Ds.append(D)
        Fs.append(feature)
        Cs.append(np.vstack([X,Y]))
        Ns.append(path.stem)

    shared_data[process_id] = {'Ws':Ws, 'Ds':Ds, 'Fs':Fs, 'Cs':Cs, 'Ns':Ns}

    print(f'process {process_id} finished. len(Ws)={len(Ws)}')

if __name__  == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="")
    ap.add_argument("-o", "--output", required=True, help="")
    ap.add_argument("-p", "--suppoint", required=True, help="")
    ap.add_argument("-l", "--ldpoint", required=True, help="")

    args = vars(ap.parse_args())
    IN_DIR  = args['input']
    OUT_DIR = args['output']
    SUPPOINT_DIR   = args['suppoint']
    LDPOINT_DIR   = args['ldpoint']

    #error_list = ['CSR2071A', 'CSR1334A', 'nl_5750a']
    error_list= ['SPRING4188', 'SPRING4100']

    DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/debug/'
    os.makedirs(DEBUG_DIR, exist_ok=True)

    cnt = 0
    #slc_ids = ['Aux_Hip_Waist_0', 'Aux_Hip_Waist_1', 'Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1', 'Aux_Waist_UnderBust_2', 'Bust', 'Aux_UnderBust_Bust_0]
    #slc_ids = ['UnderCrotch']
    slc_ids = ['Aux_Knee_UnderCrotch_2']
    failed_slice_paths = []
    for slc_id in slc_ids:
        SLICE_DIR = f'{IN_DIR}/{slc_id}/'

        DEBUG_ALIGN_DIR = f'{DEBUG_DIR}/{slc_id}_align/'
        shutil.rmtree(DEBUG_ALIGN_DIR, ignore_errors=True)
        os.makedirs(DEBUG_ALIGN_DIR, exist_ok=True)

        DEBUG_RADIAL_DIR = f'{DEBUG_DIR}/{slc_id}_radial/'
        shutil.rmtree(DEBUG_RADIAL_DIR, ignore_errors=True)
        os.makedirs(DEBUG_RADIAL_DIR, exist_ok=True)

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
        #slc_paths = slc_paths[:100]
        n_paths = len(slc_paths)
        n_processes = 12
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

        Ws = []
        Ds = []
        Fs = []
        Cs = []
        Ns = []
        print('syncronizing data across processes')
        for id, data in shared_data.items():
            n_Ws = len(data['Ws'])
            print(f'{id}, len(Ws)={n_Ws}')
            assert len(data['Ws']) > 0
            assert len(data['Ds']) > 0
            Ws.extend(data['Ws'])
            Ds.extend(data['Ds'])
            Fs.extend(data['Fs'])
            Cs.extend(data['Cs'])
            Ns.extend(data['Ns'])

        #dump all records of that slice
        n_contour = len(Ns)
        Fs = np.array(Fs)

        feature_dir_out = f'{OUT_DIR}/{slc_id}/'
        os.makedirs(feature_dir_out, exist_ok=True)

        for i in range(n_contour):
            W = Ws[i]
            D = Ds[i]
            F = Fs[i,:]
            C = Cs[i]
            name = Ns[i]
            with open(f'{feature_dir_out}{name}.pkl', 'wb') as file:
                 pickle.dump({'W':W, 'D':D, 'feature':F, 'cnt':C}, file)

    print('failed slice paths')
    print(failed_slice_paths)
    print(f'n failed slices = {len(failed_slice_paths)}')