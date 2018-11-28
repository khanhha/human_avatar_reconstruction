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
from src.caesar_slc_fix_bust import remove_arm_from_bust_slice, remove_arm_from_under_bust_slice
from scipy.spatial import ConvexHull
import scipy.ndimage as ndimage
import shutil
from scipy.interpolate import splev, splrep, splprep, splev

G_cur_file_path = Path()

def plot_segment(p0, p1, type):
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], type)

def resample_contour(X, Y, debug_path = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = 0.5 * (X[idx_ymin] + X[idx_ymax])

    X, Y = util.resample_contour(X, Y, 150)

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


def convert_contour_to_radial_code(X,Y, n_sample, path_out = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = 0.5 * (X[idx_ymin] + X[idx_ymax])
    center_y_error = Y[idx_ymax] - Y[idx_ymin]
    #print(center_y_error)
    center = np.array([center_x, center_y])

    idx_half = int(n_sample/2) + 1

    W = Y[idx_ymax] - Y[idx_ymin]
    D = X[idx_xmax] - X[idx_xmin]

    if path_out is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(X, Y, 'b-')
        plt.plot(center_x, center_y, 'r+')

    contour = LinearRing([(x,y) for x, y in zip(X,Y)])

    max_rad = 0.75*W
    angle_step = (2.0*np.pi)/float(n_sample)
    points = []
    for i in range(n_sample):
        x = np.cos(i*angle_step)
        y = np.sin(i*angle_step)
        p = center + max_rad * np.array([x,y])
        isect_ret = LineString([(center_x, center_y), (p[0],p[1])]).intersection(contour)
        if isect_ret.geom_type == 'Point':
            isect_p = np.array(isect_ret.coords[:]).flatten()
        elif isect_ret.geom_type == 'MultiPoint':
            isect_p = np.array(isect_ret[0].coords[:]).flatten()
        else:
            #assert False, 'unsupported intersection type'
            continue
        isect_p = isect_p - center
        points.append(isect_p)

    feature = []
    for i in range(1, idx_half):
        dy = points[i][1] - points[i-1][1]
        dx = points[i][0] - points[i-1][0]
        c = dy/dx
        if np.isinf(c):
         print(dx, dy)
         #assert False
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

        points_1 = util.reconstruct_slice_contour(feature, D, W)
        for i in range(points_1.shape[1]):
             p = points_1[:,i]
             p = center + p
             plt.plot(p[0], p[1], 'r+')
        plt.savefig(path_out)
        #plt.show()

    return np.array(feature), W, D

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
    #slc_ids = ['Aux_Hip_Waist_0', 'Aux_Hip_Waist_1', 'Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1', 'Aux_Waist_UnderBust_2', 'Bust']
    slc_ids = ['Bust']
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

        if slc_id == 'Aux_UnderBust_Bust_0':
            DEBUG_UNDERBUST_BUST_DIR = f'{DEBUG_DIR}/{slc_id}_aux_underbust_bust_0_cutoff/'
            shutil.rmtree(DEBUG_UNDERBUST_BUST_DIR, ignore_errors=True)
            os.makedirs(DEBUG_UNDERBUST_BUST_DIR, exist_ok=True)

        Ws = []
        Ds = []
        Fs = []
        Cs = []
        Ns = []

        for i, path in enumerate(Path(SLICE_DIR).glob('*.pkl')):
            G_cur_file_path = path

            print(path, i)

            #debug
            #if 'CSR1289A' not in path.name:
            #    continue

            with open(path, 'rb') as file:
                slc_contours = pickle.load(file)

            suppoints_path = f'{SUPPOINT_DIR}/{path.stem}.pkl'
            assert os.path.exists(suppoints_path)
            with open(suppoints_path, 'rb') as file:
                supppoints = pickle.load(file)

            ld_path = f'{LDPOINT_DIR}/{path.stem}.pkl'
            assert os.path.exists(ld_path)
            with open(ld_path, 'rb') as file:
                ld_points = pickle.load(file)

            ignore = False
            # for name in error_list:
            #     if name in  str(path.stem):
            #         ignore = True
            #         break

            if ignore:
                continue

            assert len(slc_contours) != 0

            #TODO: is the contour with the largest number of vertices the main contour of the that slice?
            lens = np.array([len(contour) for contour in slc_contours])
            contour = slc_contours[np.argmax(lens)]
            contour = np.array(contour)
            contour = contour[:, :2]

            align_anchor_pos_x = True
            if slc_id == 'Bust':
                debug_bust_path = f'{DEBUG_BUST_DIR}/{path.stem}_bust.png'
                align_anchor_pos_x = False
                arm_pnt_negx = np.array(supppoints['Bust_Arm_NegX'][:2])
                arm_pnt_posx = np.array(supppoints['Bust_Arm_PosX'][:2])
                contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_bust_slice(contour, arm_pnt_negx=arm_pnt_negx, arm_pnt_posx=arm_pnt_posx, ld_points=ld_points, debug_path=debug_bust_path)
                if has_left != fixed_left or has_right != fixed_right:
                    failed_slice_paths.append(path)

            if slc_id == 'Aux_UnderBust_Bust_0':
                debug_bust_path = f'{DEBUG_UNDERBUST_BUST_DIR}/{path.stem}_bust.png'
                align_anchor_pos_x = False
                arm_pnt_negx = np.array(supppoints['Aux_UnderBust_Bust_0_NegX'][:2])
                arm_pnt_posx = np.array(supppoints['Aux_UnderBust_Bust_0_PosX'][:2])
                contour, has_left, has_right, fixed_left, fixed_right = remove_arm_from_under_bust_slice(contour, arm_pnt_negx=arm_pnt_negx, arm_pnt_posx=arm_pnt_posx, debug_path=debug_bust_path)
                if has_left != fixed_left or has_right != fixed_right:
                    failed_slice_paths.append(path)

            #transpose, swap X and Y to make the coordinate system more natural to the contour shape
            Y = contour[:, 0]
            X = contour[:, 1]

            X, Y = util.smooth_contour(X,Y, sigma=1.0)

            debug_align_path = f'{DEBUG_ALIGN_DIR}/{path.stem}.png'
            X, Y = util.align_contour(X, Y, anchor_pos_x= align_anchor_pos_x, debug_path=debug_align_path)
            if X is None or Y is None:
                failed_slice_paths.append(path)
                continue

            X, Y = resample_contour(X, Y)

            debug_path_out = f'{DEBUG_RADIAL_DIR}/{path.stem}.png'
            feature, W, D = convert_contour_to_radial_code(X, Y, 16, path_out=debug_path_out)

            #acculumate one more slice record
            Ws.append(W)
            Ds.append(D)
            Fs.append(feature)
            Cs.append(np.vstack([X,Y]))
            Ns.append(path.stem)

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