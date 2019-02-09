import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint, LinearRing, LineString
from shapely.ops import nearest_points
import shapely.affinity as affinity
import math
from numpy.linalg import norm
from pathlib import Path

from scipy.ndimage import filters
def smooth_contour(X, Y, sigma=3):
    X_1 = filters.gaussian_filter1d(X, sigma=sigma)
    Y_1 = filters.gaussian_filter1d(Y, sigma=sigma)
    return X_1, Y_1

def is_leg_contour(name):
    leg_names = ['UnderCrotch', 'Aux_Knee_UnderCrotch_0', 'Aux_Knee_UnderCrotch_1', 'Aux_Knee_UnderCrotch_2', 'Aux_Knee_UnderCrotch_3', 'Knee', 'Calf', 'Ankle']
    if name in leg_names:
        return True
    else:
        return False

def is_torso_contour(name):
    torso_slc_ids = {'Crotch', 'Aux_Crotch_Hip_0', 'Aux_Crotch_Hip_1', 'Aux_Crotch_Hip_2',
                     'Hip', 'Aux_Hip_Waist_0', 'Aux_Hip_Waist_1' ,'Waist',
                     'Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1', 'Aux_Waist_UnderBust_2'
                     'UnderBust',
                     'Aux_UnderBust_Bust_0', 'Bust', 'Armscye', 'Aux_Armscye_Shoulder_0', 'Shoulder'}
    if name in torso_slc_ids:
        return True
    else:
        return False

def contour_center(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymin]
    return np.array([center_x, center_y])

from scipy.interpolate import splprep, splev
def resample_contour(X, Y, n_point):
    tck, u = splprep([X, Y], s=0)
    u_1 = np.linspace(0.0, 1.0, n_point)
    X, Y = splev(u_1, tck)
    return X, Y

def sample_contour_radial(X, Y, center, n_sample):
    contour = LinearRing([(x,y) for x, y in zip(X,Y)])
    range_x =  np.max(X) - np.min(X)
    range_y =  np.max(Y) - np.min(Y)
    extend_len = 2.0*max(range_x, range_y)
    angle_step = (2.0*np.pi)/float(n_sample)
    points = []
    for i in range(n_sample):
        x = np.cos(i*angle_step)
        y = np.sin(i*angle_step)
        p = center + extend_len * np.array([x,y])

        isect_ret = LineString([(center[0], center[1]), (p[0],p[1])]).intersection(contour)
        if isect_ret.geom_type == 'Point':
            isect_p = np.array(isect_ret.coords[:]).flatten()
        #elif isect_ret.geom_type == 'MultiPoint':
        #    isect_p = np.array(isect_ret[0].coords[:]).flatten()
        else:
            #assert False, 'unsupported intersection type'
            raise Exception(f'unsupported intersection type {isect_ret.geom_type}')
        isect_p = isect_p - center
        points.append(isect_p)

    return points

from scipy.fftpack import fft2, ifft2, fft, ifft, dct, idct
def calc_fourier_descriptor(X, Y, resolution, use_radial = False, path_debug = None):
    np.set_printoptions(suppress=True)
    cnt_complex = np.array([np.complex(x,y) for x, y in zip(X,Y)])
    #cnt_complex = cnt_complex[:int(cnt_complex.shape[0]/2)]
    half = int(resolution/2)
    tf_0 = fft(cnt_complex)
    if resolution % 2 == 0:
        tf_1 = np.concatenate([tf_0[0:half], tf_0[-half:]])
    else:
        tf_1 = np.concatenate([tf_0[0:half+1], tf_0[-half:]])

    #db_dif = tf_1[0:half] - tf_1[-half:]
    #print(db_dif)
    if path_debug is not None:
        res_contour = ifft(tf_1)

        res_contour = np.concatenate([np.real(res_contour).reshape(-1,1), np.imag(res_contour).reshape(-1, 1)], axis=1)

        res_range_x = np.max(res_contour[:,0]) - np.min(res_contour[:, 0])
        range_x = np.max(X) - np.min(X)
        scale_x = range_x / res_range_x

        res_range_y = np.max(res_contour[:,1]) - np.min(res_contour[:,1])
        range_y = np.max(Y) - np.min(Y)
        scale_y = range_y / res_range_y

        res_contour *= max(scale_x, scale_y)

        plt.clf()
        plt.axes().set_aspect(1.0)
        plt.plot(X, Y, '-b')
        plt.plot(X[:2], Y[:2], '+b')
        plt.plot(res_contour[:,0], res_contour[:,1], '-r')
        plt.plot(res_contour[:,0], res_contour[:,1], '+r', ms=7)
        plt.plot(res_contour[0,0], res_contour[0,1], '+r', ms=10)
        plt.plot(res_contour[1,0], res_contour[1,1], '+g', ms=10)
        plt.savefig(path_debug)
        # plt.show()
        pass

    #normalize
    tf_1 = tf_1 / norm(tf_1[1])
    #cut off the center
    fcode = []
    tf = tf_1[1:]
    if use_radial:
        angle = np.angle(tf)
        value = np.absolute(tf)
        for i in range(tf.shape[0]):
            fcode.append(angle[i])
            fcode.append(value[i])
    else:
        for i in range(tf.shape[0]):
            t = tf[i]
            fcode.append(np.real(t))
            fcode.append(np.imag(t))

    return np.array(fcode)


from scipy.fftpack import ifft
def reconstruct_contour_fourier(fourier, use_radal = False):
    n = len(fourier)
    fouriers = []
    fouriers.append(np.complex(0.0, 0.0))
    if use_radal:
        for i in range(0, n, 2):
            angle = fourier[i]
            value =  fourier[i+1]
            cpl   = value * np.exp(np.complex(re=0.0, im=angle))
            fouriers.append(cpl)
    else:
        for i in range(0, n, 2):
            real = fourier[i]
            img =  fourier[i+1]
            fouriers.append(np.complex(real, img))

    cpl_contour = ifft(fouriers)
    X = np.real(cpl_contour)
    Y = np.imag(cpl_contour)
    return np.concatenate([X.reshape(1, -1), Y.reshape(1, -1)], axis=0)

#this function is just for the debugging about the behavior of fourier descriptor
def reconstruct_contour_fourier_zero_padding(fourier, use_radal=False):
    n = len(fourier)
    fouriers = []
    fouriers.append(np.complex(0.0, 0.0))
    if use_radal:
        for i in range(0, n, 2):
            angle = fourier[i]
            value = fourier[i + 1]
            cpl = value * np.exp(np.complex(real=0.0, imag=angle))
            fouriers.append(cpl)
    else:
        for i in range(0, n, 2):
            real = fourier[i]
            img = fourier[i + 1]
            fouriers.append(np.complex(real, img))

    cpl_contour = ifft(fouriers)
    X = np.real(cpl_contour)
    Y = np.imag(cpl_contour)

    fouriers_1 = fouriers[:8] + 12*[np.complex(0, 0)] + fouriers[8:] + 12*[np.complex(0, 0)]
    cpl_contour1 = ifft(fouriers_1)
    X1 = np.real(cpl_contour1)
    Y1 = np.imag(cpl_contour1)

    plt.axes().set_aspect(1.0)
    plt.plot(X, Y, 'r')
    plt.plot(X, Y, '+r')
    plt.plot(X1, Y1, 'b')
    plt.plot(X1, Y1, '+b')
    plt.show()

    return np.concatenate([X.reshape(1, -1), Y.reshape(1, -1)], axis=0)


def reconstruct_leg_slice_contour(feature, D, W):
    p0 = np.array([D*feature[-2]*feature[-1], 0])
    n_points = len(feature) -1
    prev_p = p0
    points = []
    points.append(prev_p)

    angle_step = 2.0*np.pi / float(n_points)
    for i in range(1, n_points):
        angle = float(i)*angle_step

        l0_p0 = prev_p
        l0_p1 = prev_p + np.array([1.0, feature[i-1]])

        l1_p0 = np.array([0.0, 0.0])
        l1_p1 = np.array([np.cos(angle), np.sin(angle)])

        isct = isect_line_line(l0_p0, l0_p1, l1_p0, l1_p1)

        points.append(isct)

        prev_p = isct

    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])

    return np.vstack([X, Y])

def reconstruct_torso_slice_contour(feature, D, W, mirror = False):
    p0 = np.array([D*feature[-2]*feature[-1], 0])
    n_points = len(feature) - 2
    half_idx = int(np.ceil(n_points / 2.0))
    assert half_idx == 4 or half_idx == 5

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
        if i == half_idx:
            l1_p1 = np.array([0, W/2])
        elif i < half_idx:
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

    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])

    if mirror == True:
        X_mirror = X[1:-1][::-1]
        Y_mirror = -Y[1:-1][::-1]

        X = np.concatenate([X,X_mirror], axis=0)
        Y = np.concatenate([Y,Y_mirror], axis=0)

    return np.vstack([X, Y])

def align_torso_contour(X, Y, anchor_pos_x = True, debug_path = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymin]

    if debug_path is not None:
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
        if (anchor_pos_x == True and c[0] > center_x) or (anchor_pos_x == False and c[0] < center_x):
            if  ymin <= center_y and center_y <= ymax:
                anchor_p1 = p0
                anchor_p2 = p1
                if anchor_p1[1] > anchor_p2[1]:
                    anchor_p1, anchor_p2 = anchor_p2, anchor_p1

    if anchor_p1 is None or anchor_p2 is None:
        return None, None

    dir_1 = anchor_p2 - anchor_p1
    anchor_dir = dir_1 / norm(dir_1)
    anchor_dir[1] = abs(anchor_dir[1])

    #rotate the contour to align the anchor line
    angle = math.acos(np.dot(anchor_dir, np.array([0, 1])))
    if anchor_dir[0] < 0:
        angle = -angle

    contour_aligned = affinity.rotate(contour, angle = angle, origin=Point(anchor_p1), use_radians=True)
    X_algn = [p[0] for p in contour_aligned.exterior.coords]
    Y_algn = [p[1] for p in contour_aligned.exterior.coords]

    if debug_path is not None:
        plt.plot(anchor_p1[0], anchor_p1[1], 'r+', ms=14)
        plt.plot(anchor_p2[0], anchor_p2[1], 'r+', ms=14)
        plt.plot([anchor_p1[0], anchor_p2[0]], [anchor_p1[1], anchor_p2[1]], 'r-', ms=14)

        plt.plot(X_algn, Y_algn, 'r-')
        plt.savefig(debug_path)
        #plt.show()

    return np.array(X_algn), np.array(Y_algn)



def load_bad_slice_names(DIR, slc_id):
    txt_path = None
    for path in Path(DIR).glob('*.*'):
        if slc_id == path.stem:
            txt_path = path
            break

    if txt_path is None:
        print(f'\tno bad slice path of slice {slc_id}')
        return ()
    else:
        names = set()
        with open(str(txt_path), 'r') as file:
            for name in file.readlines():
                name = name.replace('\n','')
                names.add(name)
        return names
