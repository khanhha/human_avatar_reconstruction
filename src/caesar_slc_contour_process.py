import numpy as np
import pickle
import argparse
from pathlib import Path

if __name__  == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    args = vars(ap.parse_args())
    IN_DIR  = args['input']
    for path in Path(IN_DIR).glob('*.pkl'):
        with open(path, 'rb') as file:
            slc_contours = pickle.load(file)
        for id, contours in slc_contours.items():
            print(id, contours)
