import numpy as np
import matplotlib.pylab as plt
import argparse
from pathlib import Path
import pickle
from scipy.fftpack import ifft

def is_leg_contour(name):
    leg_names = ['UnderCrotch', 'Aux_Knee_UnderCrotch_0', 'Aux_Knee_UnderCrotch_1', 'Aux_Knee_UnderCrotch_2', 'Aux_Knee_UnderCrotch_3', 'Knee', 'Calf', 'Ankle']
    if name in leg_names:
        return True
    else:
        return False

def isect_line_line(p1, p2, p3, p4):
    a = (p1[0]-p3[0])*(p3[1]-p4[1]) - (p1[1]-p3[1])*(p3[0]-p4[0])
    b = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])
    t = a/b
    p = np.array([p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1])])
    return p

def reconstruct_contour_fourier(fourier):
    n = len(fourier)
    fouriers = []
    fouriers.append(np.complex(0.0, 0.0))
    for i in range(0, n, 2):
        real = fourier[i]
        img =  fourier[i+1]
        fouriers.append(np.complex(real, img))

    cpl_contour = ifft(fouriers)
    X = np.real(cpl_contour)
    Y = np.imag(cpl_contour)
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--IN_DIR", required=True, help="dataset input directory")
    ap.add_argument("-s", "--SLC_TYPE", required=True, help="slice type")

    args = vars(ap.parse_args())
    IN_DIR = args['IN_DIR']
    slc_id  = args['SLC_TYPE']

    PRE_DIR = f'{IN_DIR}/caesar_obj_slices/{slc_id}'
    POST_DIR = f'{IN_DIR}/caesar_obj_slices_post_process/{slc_id}'
    FONT_SIZE = 7

    all_names = [path.name for path in Path(PRE_DIR).glob('*.pkl')]
    name = all_names[np.random.choice(len(all_names))]
    for name in all_names:
        with open(f'{PRE_DIR}/{name}', 'rb') as file:
            all_contours = pickle.load(file)

        with open(f'{POST_DIR}/{name}', 'rb') as file:
            data = pickle.load(file)
            W  = data['W']
            D  = data['D']
            F  = data['feature'] #radial code as in the paper ""
            Fr = data['fourier'] #another contour code: truncated fourier.
            post_contour = data['cnt']

        if is_leg_contour(slc_id):
            f_contour = reconstruct_leg_slice_contour(F, D, W)
        else:
            f_contour = reconstruct_torso_slice_contour(F, D, W, mirror=True)

        fr_contour = reconstruct_contour_fourier(Fr)

        fig = plt.figure()
        fig.suptitle(f'{slc_id} - {name[:-4]}')
        ax = fig.add_subplot(2,2,1, aspect=1.0)
        for contour in all_contours:
            contour = np.array(contour)
            ax.plot(contour[:,0], contour[:,1], '-b')
        ax.set_title('origin contours cut \nfrom mesh', fontsize=FONT_SIZE)
        ax.set_yticklabels([]), ax.set_xticklabels([])

        ax = fig.add_subplot(2,2,2, aspect=1.0)
        ax.plot(post_contour[0,:], post_contour[1,:], '-b')
        ax.set_title('post processed contour', fontsize=FONT_SIZE)
        ax.set_yticklabels([]), ax.set_xticklabels([])

        ax = fig.add_subplot(2,2,3, aspect=1.0)
        ax.plot(f_contour[0,:], f_contour[1,:], '-b')
        ax.set_title('contour reconstructed \n from radial code', fontsize=FONT_SIZE)
        ax.set_yticklabels([]), ax.set_xticklabels([])

        ax = fig.add_subplot(2,2,4, aspect=1.0)
        ax.plot(fr_contour[0,:], fr_contour[1,:], '-b')
        ax.set_title('contour reconstructed \n from fourier descriptor', fontsize=FONT_SIZE)
        ax.set_yticklabels([]), ax.set_xticklabels([])

        plt.show()
        plt.close('all')