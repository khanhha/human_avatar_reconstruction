import pickle
from pathlib import Path
import sys
import numpy as np

class SlcData():
    def __init__(self, id, fnames, contours, features, W, D):
        self.id = id
        self.fnames = fnames
        self.fnames_map = {name:idx for idx, name in enumerate(self.fnames)}
        self.contours = contours
        self.features = features
        self.W = W
        self.D = D

    def __getitem__(self, slc_id):
        idx = self.fnames_map[slc_id]
        x = self.W[idx]/self.D[idx]
        y = self.features[idx]
        return x, y

    def __len__(self):
        return len(self.fnames)

    #minimum set of file names shared by all slices
    @staticmethod
    def extract_shared_fnames(slices):
        shared_fnames = set()
        for slc_data in slices:
            if len(shared_fnames) == 0:
                shared_fnames = {name for name in slc_data.fnames}
            shared_fnames = shared_fnames.intersection(slc_data.fnames)

        return shared_fnames

    @staticmethod
    def build_training_data(in_slices, out_slc, fnames = None):
        if fnames is None:
            shared_fnames = SlcData.extract_shared_fnames(in_slices + [out_slc])
        else:
            shared_fnames = fnames

        #extract X and Y for the list of file names
        X = []
        Y = []
        for fname in shared_fnames:
            cur_x = []
            for slc_data in in_slices:
                x, _ = slc_data[fname]
                cur_x.append(x)
            X.append(cur_x)

            _, y = out_slc[fname]
            Y.append(y)

        return np.array(X), np.array(Y)

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

            if np.isinf(feature).flatten().sum() > 0:
                print(f'inf feature: {path}', file=sys.stderr)
                continue

            slc_names.append(path.stem)

            #W and D arrays are just of the sake of test inference
            W.append(w)
            D.append(d)

            X.append(w / d)
            Y.append(feature)

    #print_statistic(X, Y)

    return np.array(X), np.array(Y), W, D, slc_names

def load_slice_data_1(id, SLC_CONTOUR_DIR, SLC_FEATURE_DIR, bad_slc_names):
    slc_names = []
    features =  []
    W, D = [], []
    contours = []

    all_paths = [path for path in Path(SLC_FEATURE_DIR).glob('*.*')]
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

            if np.isinf(feature).flatten().sum() > 0:
                print(f'inf feature: {path}', file=sys.stderr)
                continue

            contour_path = f'{SLC_CONTOUR_DIR}/{path.name}'
            with open(str(contour_path ), 'rb') as file:
                record = pickle.load(file)
                cnt = record['cnt']
                contours.append(cnt)

            slc_names.append(path.stem)
            W.append(w)
            D.append(d)
            features.append(feature)

    return SlcData(id=id, fnames=slc_names, contours=contours, features=features, W=W, D=D)


def load_slc_contours(SLC_DIR):
    contours = {}
    for path in Path(SLC_DIR).glob('*.pkl'):
        with open(str(path), 'rb') as file:
            record = pickle.load(file)
            contours[path.stem] = record['cnt']
    return contours