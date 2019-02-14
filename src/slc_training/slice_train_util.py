import pickle
from pathlib import Path
import sys
import numpy as np

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

def load_slice_data(SLC_CODE_DIR, bad_slc_names):
    slc_names = []
    all_paths = [path for path in Path(SLC_CODE_DIR).glob('*.*')]
    X, Y = [], []
    W, D = [], []
    for path in all_paths:
        if path.stem in bad_slc_names:
            continue
        with open(str(path), 'rb') as file:
            record = pickle.load(file)
            w = record['W']
            d = record['D']

            if w == 0.0 or d == 0.0:
                print('zero w or d: ', w, d, file=sys.stderr)
                continue

            feature = record['Code']

            if np.isnan(feature).flatten().sum() > 0:
                print(f'nan feature: {path}', file=sys.stderr)
                continue

            if np.isnan(X).flatten().sum() > 0:
                print(f'nan X: {path}', file=sys.stderr)
                continue

            if np.isinf(feature).flatten().sum() > 0:
                print(f'inf feature: {path}', file=sys.stderr)
                continue

            if np.isinf(X).flatten().sum() > 0:
                print(f'inf X: {path}', file=sys.stderr)
                continue

            slc_names.append(path.stem)

            #W and D arrays are just of the sake of test inference
            W.append(w)
            D.append(d)

            X.append(w / d)
            Y.append(feature)

    #print_statistic(X, Y)

    return np.array(X), np.array(Y), W, D, slc_names

def load_slc_contours(SLC_DIR):
    contours = {}
    for path in Path(SLC_DIR).glob('*.pkl'):
        with open(str(path), 'rb') as file:
            record = pickle.load(file)
            contours[path.stem] = record['cnt']
    return contours