import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import join
import os
import shutil
import cv2 as cv
from pca.nn_util import crop_silhouette_pair
from tqdm import tqdm
import matplotlib.pyplot as plt

def copy_files(paths, out_dir, args):
    for path in Path(out_dir).glob('*.*'):
        os.remove(str(path))

    os.makedirs(out_dir, exist_ok=True)
    for path in tqdm(paths, desc=f'copied files to {out_dir}'):
        out_name = str(path.stem).replace('_front','')
        out_name = out_name.replace('_side', '')
        shutil.copy(src=path, dst=join(*[out_dir, f'{out_name}.jpg']))

def copy_train_valid_test(name_ids, id_to_path_dict, train_idxs, valid_idxs, test_idxs, out_dir, args):
    train_dir = join(*[out_dir, 'train'])
    valid_dir = join(*[out_dir, 'valid'])
    test_dir  = join(*[out_dir, 'test'])

    copy_files([id_to_path_dict[name_ids[idx]] for idx in train_idxs], train_dir, args)
    copy_files([id_to_path_dict[name_ids[idx]] for idx in valid_idxs], valid_dir, args)
    copy_files([id_to_path_dict[name_ids[idx]] for idx in test_idxs], test_dir, args)

def crop_pairs(sil_f_dir, sil_s_dir, size):
    fpaths = sorted([path for path in Path(sil_f_dir).glob('*.*')])
    spaths = sorted([path for path in Path(sil_s_dir).glob('*.*')])
    for fpath, spath in zip(fpaths, spaths):
        assert fpath.name == spath.name

    path_pairs = [(fpath, spath) for fpath, spath in zip(fpaths, spaths)]

    for fpath, spath in tqdm(path_pairs , desc=f'crop silhouette pairs {Path(sil_f_dir).name} - {Path(sil_s_dir).name}'):
        sil_f = cv.imread(str(fpath), cv.IMREAD_GRAYSCALE)
        sil_s = cv.imread(str(spath), cv.IMREAD_GRAYSCALE)

        th3, sil_f  = cv.threshold(sil_f, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        th3, sil_s  = cv.threshold(sil_s, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        sil_f, sil_s = crop_silhouette_pair(sil_f, sil_s, mask_f=sil_f, mask_s=sil_s, target_h=size[0], target_w=size[1], px_height=int(0.9*size[0]))

        th3, sil_f  = cv.threshold(sil_f, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        th3, sil_s  = cv.threshold(sil_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # plt.subplot(121), plt.imshow(sil_f)
        # plt.subplot(122), plt.imshow(sil_s)
        # plt.show()

        cv.imwrite(str(fpath), img=sil_f)
        cv.imwrite(str(spath), img=sil_s)

def crop_train_test_valid(base_sil_f_dir, base_sil_s_dir, size):
    names = ['train', 'test', 'valid']
    for name in names:
        f_dir = join(*[base_sil_f_dir, name])
        s_dir = join(*[base_sil_s_dir, name])
        crop_pairs(f_dir, s_dir, size)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir",  default=True, required=False)
    ap.add_argument("-sil_s_dir",  default=True, required=False)
    ap.add_argument("-out_dir", default=False, required=False)
    ap.add_argument("-resize_size", type=str,  default='360x360', required=False)
    ap.add_argument("-post_process",  action='store_true')
    args = ap.parse_args()

    size = args.resize_size.split('x')
    size = (int(size[0]), int(size[1]))

    sil_f_paths = dict([(path.stem, path) for path in Path(args.sil_f_dir).glob('*.*')])
    sil_s_paths = dict([(path.stem, path) for path in Path(args.sil_s_dir).glob('*.*')])
    assert sil_f_paths.keys() == sil_s_paths.keys()

    n = len(sil_f_paths)

    name_ids = [id for id in sil_s_paths.keys()]

    sil_f_dir = join(*[args.out_dir, 'sil_f'])
    sil_s_dir = join(*[args.out_dir, 'sil_s'])
    os.makedirs(sil_f_dir, exist_ok=True)
    os.makedirs(sil_s_dir, exist_ok=True)

    np.random.seed(100)
    train_idxs, test_idxs  = train_test_split(np.arange(n), test_size=0.05)
    train_idxs, valid_idxs = train_test_split(train_idxs, test_size=0.1)
    copy_train_valid_test(name_ids, sil_f_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=sil_f_dir, args=args)
    copy_train_valid_test(name_ids, sil_s_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=sil_s_dir, args=args)

    crop_train_test_valid(sil_f_dir, sil_s_dir, size)