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
from src.util import smooth_contour
import src.util as util
from scipy.spatial import ConvexHull
import scipy.ndimage as ndimage
import shutil
from scipy.interpolate import splev, splrep, splprep, splev

G_cur_file_path = Path()


def plot_segment(p0, p1, type):
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], type)

def resample_contour(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = 0.5 * (X[idx_ymin] + X[idx_ymax])

    tck, u = splprep([X, Y], s=0)
    u_1 = np.linspace(0.0, 1.0, 150)
    X, Y = splev(u_1, tck)

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

    # okay = np.where(np.abs(np.diff(X)) + np.abs(np.diff(Y)) > 0)
    # X = np.r_[X[okay], X[-1], X[0]]
    # Y = np.r_[Y[okay], Y[-1], Y[0]]
    #
    # #fix bad points
    # jump = np.sqrt(np.diff(X) ** 2 + np.diff(X) ** 2)
    # smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
    # limit = 2 * np.median(smooth_jump)  # factor 2 is arbitrary
    # xn, yn = X[:-1], Y[:-1]
    # X = xn[(jump > 0) & (smooth_jump < limit)]
    # Y = yn[(jump > 0) & (smooth_jump < limit)]
    #
    # #resample
    # tck, u = splprep([X, Y], s=0)
    # u_1 = np.linspace(0.0, 1.0, 150)
    # X, Y = splev(u_1, tck)

    # plt.clf()
    # plt.axes().set_aspect(1)
    # plt.plot(X, Y, 'b+')
    # plt.plot(center_x, center_y, 'r+')
    # plt.plot(X[0], Y[0], 'r+', ms = 20)
    # plt.show()
    return X, Y

def align_contour(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymin]

    DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/hip_hull/'

    plt.clf()
    plt.axes().set_aspect(1)
    plt.plot(X, Y, 'b+')
    plt.plot(center_x, center_y, 'r+')

    contour = Polygon([(x,y) for x, y in zip(X,Y)])
    contour_1 = contour.simplify(0.01, preserve_topology=False)
    contour_2 = contour_1.convex_hull
    for p in contour_2.exterior.coords:
        plt.plot(p[0], p[1], 'r+', ms=7)

    #find the anchor segment
    n_point = len(contour_2.exterior.coords)
    anchor_p1 = None
    anchor_p2 = None
    for i in range(n_point):
        p0 = np.array(contour_2.exterior.coords[i])
        p1 = np.array(contour_2.exterior.coords[(i+1)%n_point])
        c = 0.5*(p0+p1)
        ymin = min(p0[1], p1[1])
        ymax = max(p0[1], p1[1])
        if c[0] > center_x and ymin <= center_y and center_y <= ymax:
            anchor_p1 = p0
            anchor_p2 = p1
            if anchor_p1[1] > anchor_p2[1]:
                anchor_p1, anchor_p2 = anchor_p2, anchor_p1

    dir_1 = anchor_p2 - anchor_p1
    anchor_dir = dir_1 / norm(dir_1)
    anchor_dir[1] = abs(anchor_dir[1])

    #rotate the contour to align the anchor line
    angle = math.acos(np.dot(anchor_dir, np.array([0, 1])))
    if anchor_dir[0] < 0:
        angle = -angle

    contour_aligned = affinity.rotate(contour, angle = angle, origin=Point(anchor_p1), use_radians=True)

    #plt.plot(anchor_p0[0], anchor_p0[1], 'r+', ms=14)
    plt.plot(anchor_p1[0], anchor_p1[1], 'r+', ms=14)
    plt.plot(anchor_p2[0], anchor_p2[1], 'r+', ms=14)
    #plt.plot(anchor_p3[0], anchor_p3[1], 'r+', ms=14)

    X_algn = [p[0] for p in contour_aligned.exterior.coords]
    Y_algn = [p[1] for p in contour_aligned.exterior.coords]
    plt.plot(X_algn, Y_algn, 'r-')
    plt.savefig(f'{DEBUG_DIR}{G_cur_file_path.stem}.png')
    #plt.show()

    return X_algn, Y_algn

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

        points_1 = util.reconstruct_slice_contour(feature, D, D)
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

    args = vars(ap.parse_args())
    IN_DIR  = args['input']
    OUT_DIR = args['output']

    DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/radial_code/'
    #error_list = ['CSR2071A', 'CSR1334A', 'nl_5750a']
    error_list= ['SPRING4188', 'SPRING4100']

    Ws = []
    Ds = []
    Fs = []
    Cs = []
    Ns = []

    cnt = 0

    for i, path in enumerate(Path(IN_DIR).glob('*.pkl')):

        # cnt += 1
        # if cnt > 10:
        #     break

        G_cur_file_path = path
        #if 'SPRING1413' not in str(path):
        #    continue
        print(path, i)

        with open(path, 'rb') as file:
            slc_contours = pickle.load(file)

        ignore = False
        # for name in error_list:
        #     if name in  str(path.stem):
        #         ignore = True
        #         break

        if ignore:
            continue

        for id, contours in slc_contours.items():
            if id != 'Hip':
                continue

            lens = np.array([len(contour) for contour in contours])
            contour = contours[np.argmax(lens)]
            #transpose, swap X and Y to make the coordinate system more natural to the contour shape
            Y = [p[0] for p in contour]
            X = [p[1] for p in contour]
            os.makedirs(f'{DEBUG_DIR}{id}/', exist_ok=True)
            debug_path_out = f'{DEBUG_DIR}{id}/{path.stem}.png'

            X, Y = smooth_contour(X,Y, sigma=1.0)
            X, Y = align_contour(X, Y)
            X, Y = resample_contour(X, Y)
            feature, W, D = convert_contour_to_radial_code(X, Y, 16, path_out=debug_path_out)

            #
            #feature_dir_out = f'{OUT_DIR}{id}/'
            #os.makedirs(feature_dir_out, exist_ok=True)

            #data[path.stem] = {'W':W, 'D':D, 'feature':feature, 'cnt':np.vstack([X,Y])}
            Ws.append(W)
            Ds.append(D)
            Fs.append(feature)
            Cs.append(np.vstack([X,Y]))
            Ns.append(path.stem)

            #with open(f'{feature_dir_out}{path.stem}.pkl', 'wb') as file:
            #     pickle.dump({'W':W, 'D':D, 'feature':feature, 'cnt':np.vstack([X,Y])}, file)

    n_contour = len(Ns)
    Fs = np.array(Fs)

    #fix the first curvature
    curvs_0 = Fs[:, 0]
    inf_mask = np.isinf(curvs_0)
    print('inf value count: ', np.sum(inf_mask))
    curvs_0 = curvs_0[~inf_mask]
    curvs_0_mean   = np.mean(curvs_0)
    curvs_0_median = np.median(curvs_0)
    print("mean: ", curvs_0_mean, ' median: ', curvs_0_median)
    Fs[:, 0] = curvs_0_median
    #Fs[inf_mask, 0] = curvs_0_median

    #fix the last curvature
    curvs_0 = Fs[:, -3]
    inf_mask = np.isinf(curvs_0)
    print('inf value count: ', np.sum(inf_mask))
    curvs_0 = curvs_0[~inf_mask]
    curvs_0_mean   = np.mean(curvs_0)
    curvs_0_median = np.median(curvs_0)
    print("mean: ", curvs_0_mean, ' median: ', curvs_0_median)
    Fs[:, -3] = curvs_0_median
    #Fs[inf_mask, -3] = curvs_0_median

    feature_dir_out = f'{OUT_DIR}/Hip/'
    os.makedirs(feature_dir_out, exist_ok=True)

    for i in range(n_contour):
        W = Ws[i]
        D = Ds[i]
        F = Fs[i,:]
        C = Cs[i]
        name = Ns[i]
        with open(f'{feature_dir_out}{name}.pkl', 'wb') as file:
             pickle.dump({'W':W, 'D':D, 'feature':F, 'cnt':C}, file)