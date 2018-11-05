import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path

if __name__  == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    args = vars(ap.parse_args())
    IN_DIR  = args['input']

    DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/debug/'
    for path in Path(IN_DIR).glob('*.pkl'):
        print(path)
        with open(path, 'rb') as file:
            slc_contours = pickle.load(file)

        for id, contours in slc_contours.items():
            lens = np.array([len(contour) for contour in contours])
            contour = contours[np.argmax(lens)]
            X = [p[0] for p in contour]
            Y = [p[1] for p in contour]
            plt.clf()
            plt.plot(X,Y)
            plt.title(id)
            #plt.show()
            os.makedirs(f'{DEBUG_DIR}{id}/', exist_ok=True)
            plt.savefig(f'{DEBUG_DIR}{id}/{path.stem}.png')