import numpy as np
import matplotlib.pylab as plt
import argparse
from pathlib import Path
import pickle
import src.util as  util

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

        if util.is_leg_contour(slc_id):
            f_contour = util.reconstruct_leg_slice_contour(F, D, W)
        else:
            f_contour = util.reconstruct_torso_slice_contour(F, D, W, mirror=True)

        fr_contour = util.reconstruct_contour_fourier(Fr)

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