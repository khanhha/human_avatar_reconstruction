import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import join
import os
import shutil

def copy_files(paths, out_dir):
    for path in Path(out_dir).glob('*.*'):
        os.remove(str(path))

    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        out_name = str(path.stem).replace('_front','')
        out_name = out_name.replace('_side', '')
        shutil.copy(src=path, dst=join(*[out_dir, f'{out_name}.png']))

def copy_train_valid_test(name_ids, id_to_path_dict, train_idxs, valid_idxs, test_idxs, out_dir):
    train_dir = join(*[out_dir, 'train'])
    valid_dir = join(*[out_dir, 'valid'])
    test_dir  = join(*[out_dir, 'test'])

    copy_files([id_to_path_dict[name_ids[idx]] for idx in train_idxs], train_dir)
    copy_files([id_to_path_dict[name_ids[idx]] for idx in valid_idxs], valid_dir)
    copy_files([id_to_path_dict[name_ids[idx]] for idx in test_idxs], test_dir)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-in_dir",  default=True, required=False)
    ap.add_argument("-out_dir", default=False, required=False)
    args = ap.parse_args()

    sil_f_paths = dict([(path.stem.replace('_front', ''), path) for path in Path(args.in_dir).glob('*front.png')])
    sil_s_paths = dict([(path.stem.replace('_side', ''),  path) for path in Path(args.in_dir).glob('*side.png')])
    assert sil_f_paths.keys() == sil_s_paths.keys()

    n = len(sil_f_paths)

    name_ids = [id for id in sil_s_paths.keys()]

    train_idxs, test_idxs  = train_test_split(np.arange(n), test_size=0.05)
    train_idxs, valid_idxs = train_test_split(train_idxs, test_size=0.1)

    sil_f_dir = join(*[args.out_dir, 'sil_f'])
    os.makedirs(sil_f_dir, exist_ok=True)
    copy_train_valid_test(name_ids, sil_f_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=sil_f_dir)

    sil_s_dir = join(*[args.out_dir, 'sil_s'])
    os.makedirs(sil_s_dir, exist_ok=True)
    copy_train_valid_test(name_ids, sil_s_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=sil_s_dir)

