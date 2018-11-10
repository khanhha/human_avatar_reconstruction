import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing, LineString
from pathlib import Path

def isect_line_line(p1, p2, p3, p4):
    a = (p1[0]-p3[0])*(p3[1]-p4[1]) - (p1[1]-p3[1])*(p3[0]-p4[0])
    b = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])
    t = a/b
    p = np.array([p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1])])
    return p

def reconstruct_slice_contour(feature, D, W):
    p0 = np.array([D*feature[-2]*feature[-1], 0])
    n_points = len(feature) - 2
    half_idx = int(n_points / 2.0)
    assert half_idx == 4

    prev_p = p0

    points = []
    points.append(prev_p)

    angle_step = np.pi / float(n_points)
    for i in range(1, n_points+1):
        angle = float(i)*angle_step
        #TODO
        # if i == 4:
        #     x, y = 0.0, W/2.0
        # else:
        #     x = (prev_p[1] - feature[i] * prev_p[0]) / (np.tan(angle) - feature[i])
        #     y = np.tan(angle) * x
        #prev_p = np.array([x,y])
        #points.append(prev_p)

        # test
        l0_p0 = prev_p
        l0_p1 = prev_p + np.array([1.0, feature[i-1]])

        l1_p0 = np.array([0.0, 0.0])
        if i == 4:
            l1_p1 = np.array([0, W/2])
        elif i < 4:
            l1_p1 = np.array([1.0, np.tan(angle)])
        else:
            l1_p1 = np.array([-1.0, -np.tan(angle)])

        isct = isect_line_line(l0_p0, l0_p1, l1_p0, l1_p1)

        points.append(isct)

        prev_p = isct

        #print(isct[0] - x, isct[1] - y)
        #plt.plot([l0_p0[0], l0_p1[0]], [l0_p0[1], l0_p1[1]], 'r-')
        #plt.plot([l1_p0[0], l1_p1[0]], [l1_p0[1], l1_p1[1]], 'b-')
        #plt.plot(isct[0], isct[1], 'b+')

    return points

def plot_segment(p0, p1, type):
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], type)

def convert_contour_to_radial_code(X,Y, n_sample, path_out = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymin]
    center_y_error = Y[idx_ymax] - Y[idx_ymin]
    #print(center_y_error)
    center = np.array([center_x, center_y])

    idx_half = int(n_sample/2) + 1

    W = Y[idx_ymax] - Y[idx_ymin]
    D = X[idx_xmax] - X[idx_xmin]

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
         feature.append(c)

    #print(feature)

    D_mid =  np.linalg.norm(points[0] - points[idx_half])
    r1 = D_mid / D
    r2 = np.linalg.norm(points[0]) /  D_mid
    feature.extend([r1, r2])
    if path_out is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(X, Y, 'b-')
        plt.plot(center_x, center_y, 'r+')
        for p in points:
            p = center + p
            plt.plot([center_x, p[0]], [center_y, p[1]], 'r-')
            plt.plot(p[0], p[1], 'b+')

        # angle_step = np.pi / 8.0
        # prev_p = center + points[0]
        # for i in range(1, idx_half):
        #     angle = float(i)*angle_step
        #
        #     if i == 4 or i == 8:
        #         continue
        #
        #     if i <= 4:
        #         dir = np.array([1.0, np.tan(angle)])
        #     else:
        #         dir = np.array([-1.0, -np.tan(angle)])
        #
        #     p_1 = center + 1.0*dir
        #
        #     cur = feature[i-1]
        #     p_cur = prev_p - 0.1*np.array([1.0, cur])
        #
        #     prev_p = isect_line_line(center, p_1, prev_p, p_cur)
        #
        #     plt.plot(prev_p[0], prev_p[1], 'r+')
        #
        #     print(cur, dir, p_cur)

            #plt.plot([p[0], p_cur[0]], [p[1], p_cur[1]], 'r-')

        points_1 =  reconstruct_slice_contour(feature, D, D)
        for p in points_1:
             p = center + p
             plt.plot(p[0], p[1], 'r+')
        #plt.savefig(path_out)
        plt.show()

if __name__  == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    args = vars(ap.parse_args())
    IN_DIR  = args['input']

    DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/debug/'
    ignore_list = ['CSR2071A', 'CSR1334A', 'nl_5750a']
    for i, path in enumerate(Path(IN_DIR).glob('*.pkl')):
        print(path, i)
        if 'SPRING1477' not in str(path):
            continue

        with open(path, 'rb') as file:
            slc_contours = pickle.load(file)

        ignore = False
        for name in ignore_list:
            if name in  str(path.stem):
                ignore = True
                break

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
            convert_contour_to_radial_code(X,Y, 16, path_out=f'{DEBUG_DIR}{id}/{path.stem}.png')
            #                                                                                                                                                                   plt.savefig(f'{DEBUG_DIR}{id}/{path.stem}.png')