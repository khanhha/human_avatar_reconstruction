import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import join
import os
import shutil
import cv2 as cv
from pca.nn_util import crop_silhouette_pair_blender
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import tempfile
from pca.pca_vic_model import PcaModel
from sklearn.externals import joblib
import sklearn

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

def crop_a_pair(size, path_pair):
    fpath = path_pair[0]
    spath = path_pair[1]

    sil_f = cv.imread(str(fpath), cv.IMREAD_GRAYSCALE)
    sil_s = cv.imread(str(spath), cv.IMREAD_GRAYSCALE)

    sil_f, sil_s = crop_silhouette_pair_blender(sil_f, sil_s, size)

    # plt.subplot(121), plt.imshow(sil_f)
    # plt.subplot(122), plt.imshow(sil_s)
    # plt.show()

    cv.imwrite(str(fpath), img=sil_f)
    cv.imwrite(str(spath), img=sil_s)

def crop_pairs(sil_f_dir, sil_s_dir, size):
    fpaths = sorted([path for path in Path(sil_f_dir).glob('*.*')])
    spaths = sorted([path for path in Path(sil_s_dir).glob('*.*')])
    for fpath, spath in zip(fpaths, spaths):
        assert fpath.name == spath.name

    path_pairs = [(fpath, spath) for fpath, spath in zip(fpaths, spaths)]

    with Pool(10) as p:
        with tqdm(total=len(path_pairs)) as pbar:
            for i, _ in enumerate(p.imap_unordered(partial(crop_a_pair, size), path_pairs)):
                pbar.update()

def crop_train_test_valid(base_sil_f_dir, base_sil_s_dir, size):
    names = ['train', 'test', 'valid']
    for name in names:
        f_dir = join(*[base_sil_f_dir, name])
        s_dir = join(*[base_sil_s_dir, name])
        crop_pairs(f_dir, s_dir, size)

def copy_file_prefix(in_dir, out_dir, prefix, n_file = 10000):
    os.makedirs(out_dir, exist_ok=True)

    paths =list([path for path in Path(in_dir).glob('*.*')])
    #paths = sklearn.utils.shuffle(paths)

    for path in tqdm(paths, desc= f'copy_file_prefix {prefix}'):
        shutil.copy(str(path), os.path.join(*[out_dir, f'{prefix}_{path.name}']))

def copy_target_prefix(in_dir, out_dir, prefix):
    assert prefix in ['_male', '_female']
    os.makedirs(out_dir, exist_ok=True)

    ex_val = np.array([1.0]) if prefix == '_male' else np.array([0.0])

    paths = [path for path in Path(in_dir).glob('*.*')]

    for path in tqdm(paths, desc=f'copy_target_prefix {prefix}'):
        param = np.load(path)
        param = np.hstack([ex_val, param])
        np.save(os.path.join(*[out_dir, f'{prefix}_{path.name}']), param)

def remove_missing_pair(sil_f_dir, sil_s_dir):
    sil_f_names = set([path.name for path in Path(sil_f_dir).glob('*.*')])
    sil_s_names = set([path.name for path in Path(sil_s_dir).glob('*.*')])

    common_names = sil_f_names.intersection(sil_s_names)

    bad_f_names = sil_f_names.difference(common_names)
    bad_s_names = sil_s_names.difference(common_names)

    for name in bad_f_names:
        print(f'remove file {name}')
        path = os.path.join(*[sil_f_dir, name])
        os.remove(path)

    for name in bad_s_names:
        print(f'remove file {name}')
        path = os.path.join(*[sil_s_dir, name])
        os.remove(path)

def dump_heights(pca_in_dir, pca_ml_model_path, pca_fml_model_path, height_out_path):
    ml_model = joblib.load(pca_ml_model_path)
    fml_model = joblib.load(pca_fml_model_path)

    heights = []
    paths = [path for path in Path(pca_in_dir).glob('*.*')]
    for path in tqdm(paths, desc='dump height'):
        param = np.load(path)
        if '_male' in path.stem:
            assert param[0] >= 0.9, f'{param[0]} >= 0.9, {path.name}'
            verts = ml_model.inverse_transform(param[1:])
        else:
            assert param[0] <= 0.1, f'{param[0]} <= 0.1, {path.name}'
            verts = fml_model.inverse_transform(param[1:])

        verts = verts.reshape(verts.shape[0] // 3, 3)
        h = verts[:, 2].max() - verts[:, 2].min()
        heights.append((path.stem, h))

    with open(height_out_path, 'wt') as file:
        file.writelines(f"{l[0]} {l[1]}\n" for l in heights)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_ml_dir",  default=True, required=False)
    ap.add_argument("-sil_s_ml_dir",  default=True, required=False)
    ap.add_argument("-sil_f_fml_dir",  default=True, required=False)
    ap.add_argument("-sil_s_fml_dir",  default=True, required=False)

    ap.add_argument("-target_ml_dir",  default=True, required=False)
    ap.add_argument("-target_fml_dir",  default=True, required=False)

    ap.add_argument("-pca_ml_model_path",  default=True, required=False)
    ap.add_argument("-pca_fml_model_path",  default=True, required=False)

    ap.add_argument("-out_dir", default=False, required=False)
    ap.add_argument("-resize_size", type=str,  default='360x360', required=False)
    ap.add_argument("-post_process",  action='store_true')
    args = ap.parse_args()

    size = args.resize_size.split('x')
    size = (int(size[0]), int(size[1]))

    os.makedirs(args.out_dir, exist_ok=True)

    #save pca model
    model_female = joblib.load(filename=args.pca_fml_model_path)
    model_male = joblib.load(filename=args.pca_ml_model_path)
    pca_model = PcaModel(model_female=model_female, model_male=model_male)
    out_path = os.path.join(*[args.out_dir, 'pca_model.jlb'])
    pca_model.dump(out_path)
    print(f'dump pca model to {out_path}')

    # copy pca target with the same name pattern to the out dir
    out_target_dir = os.path.join(*[args.out_dir, 'target'])
    copy_target_prefix(args.target_fml_dir, out_target_dir, '_female')
    copy_target_prefix(args.target_ml_dir,  out_target_dir, '_male')

    out_height_path = os.path.join(*[args.out_dir, 'height.txt'])
    dump_heights(pca_in_dir=out_target_dir,
                 pca_ml_model_path=args.pca_ml_model_path, pca_fml_model_path=args.pca_fml_model_path,
                 height_out_path=out_height_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f'created temporary dir: {tmp_dir}')

        tmp_sil_f_dir = os.path.join(*[tmp_dir, 'sil_f'])
        tmp_sil_s_dir = os.path.join(*[tmp_dir, 'sil_s'])

        # merge both male and female to a tempt dir and make their file names distinctive
        copy_file_prefix(args.sil_f_ml_dir, tmp_sil_f_dir, '_male')
        copy_file_prefix(args.sil_s_ml_dir, tmp_sil_s_dir, '_male')

        copy_file_prefix(args.sil_f_fml_dir, tmp_sil_f_dir, '_female')
        copy_file_prefix(args.sil_s_fml_dir, tmp_sil_s_dir, '_female')

        remove_missing_pair(tmp_sil_f_dir, tmp_sil_s_dir)

        sil_f_paths = dict([(path.stem, path) for path in Path(tmp_sil_f_dir).glob('*.*')])
        sil_s_paths = dict([(path.stem, path) for path in Path(tmp_sil_s_dir).glob('*.*')])
        assert sil_f_paths.keys() == sil_s_paths.keys()

        n = len(sil_f_paths)
        n_females = len([name for name in sil_f_paths.keys() if '_female' in name])
        n_males   = len([name for name in sil_f_paths.keys() if '_male' in name])
        print(f'n females = {n_females}, n_males = {n_males}')

        name_ids = [id for id in sil_s_paths.keys()]

        out_sil_f_dir = join(*[args.out_dir, 'sil_f'])
        out_sil_s_dir = join(*[args.out_dir, 'sil_s'])
        os.makedirs(out_sil_f_dir, exist_ok=True)
        os.makedirs(out_sil_s_dir, exist_ok=True)

        np.random.seed(100)
        label = np.zeros(len(name_ids), dtype=np.uint8)
        for idx, name in enumerate(name_ids):
            label[idx] = 1 if '_male' in name else 0
        train_idxs, test_idxs  = train_test_split(np.arange(n), test_size=0.1, stratify=label) #big test size for reduced traning time

        label = np.zeros(len(train_idxs), dtype=np.uint8)
        for i in range(len(train_idxs)):
            label[i] = 1 if '_male' in name_ids[train_idxs[i]] else 0
        train_idxs, valid_idxs = train_test_split(train_idxs, test_size=0.10, stratify=label)

        copy_train_valid_test(name_ids, sil_f_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=out_sil_f_dir, args=args)
        copy_train_valid_test(name_ids, sil_s_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=out_sil_s_dir, args=args)

    print(f'deleted temporary dir')

    crop_train_test_valid(out_sil_f_dir, out_sil_s_dir, size)

