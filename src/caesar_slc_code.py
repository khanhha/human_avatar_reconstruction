import os
from pathlib import Path
import argparse
import multiprocessing
import pickle
from functools import partial
import shutil
import src.util as util

def fourier(record, n):
    contour = record['cnt']
    X = contour[0,:]
    Y = contour[1,:]
    code = util.calc_fourier_descriptor(X, Y, resolution=n, path_debug=None)
    return code

def fourier_resolution(slc_id):
    if util.is_leg_contour(slc_id):
        return 8
    elif slc_id in ['Shoulder', 'Aux_Armscye_Shoulder_0']:
        return 20
    else:
        return 16

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--IN_DIR", required=True, help="")
    ap.add_argument("-o", "--OUT_DIR", required=True, help="")
    ap.add_argument("-ids", "--slc_ids", required=True, help="")

    args = vars(ap.parse_args())

    IN_DIR = args['IN_DIR']
    OUT_DIR = args['OUT_DIR']
    ids = args['slc_ids']

    nprocess = 12
    pool = multiprocessing.Pool(12)

    all_ids = [path.stem for path in Path(IN_DIR).glob('./*')]
    if ids == 'all':
        slc_ids = all_ids
    else:
        slc_ids = ids.split(',')
        for id in slc_ids:
            assert id in all_ids, f'{id} is unrecognized'

    for slc_id in slc_ids:
        SLC_DIR = f'{IN_DIR}/{slc_id}/'
        paths = [path for path in Path(SLC_DIR).glob('*.pkl')]
        records = []
        for path in paths:
            if 'CSR1228A' in path.stem:
                debug = True
            with open(str(path), 'rb') as file:
                r = pickle.load(file)
                records.append(r)

        print(f'{slc_id} : n_slices = {len(paths)}')
        resolution = fourier_resolution(slc_id)
        codes = pool.map(func=partial(fourier, n=resolution), iterable=records, chunksize=128)

        CODE_OUT_DIR = f'{OUT_DIR}/fourier/{slc_id}'
        shutil.rmtree(CODE_OUT_DIR, ignore_errors=True)
        os.makedirs(CODE_OUT_DIR)

        for path, record, code in zip(paths, records, codes):
            W = record['W']
            D = record['D']
            with open(f'{CODE_OUT_DIR}/{path.stem}.pkl', 'wb') as file:
                pickle.dump(file=file, obj={'Code':code, 'W':W, 'D':D})



