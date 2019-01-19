import scipy.io as io
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from src.obj_util import export_mesh, export_vertices

def extract_landmark_coords(points, ld_idxs):
    points_out = []
    for idx in ld_idxs:
        idx = idx[0]
        if np.isnan(idx) or idx < 0:
            p_ld = np.array([0.0, 0.0, 0.0])
        else:
            p_ld = points[idx,:]
        points_out.append(p_ld)

    return np.array(points_out)

def convert_mat_to_obj(DIR_IN, DIR_OUT, face_mat_path, ld_idxs):
    faces = io.loadmat(face_mat_path)['faces']
    n_files = len([path for path in Path(DIR_IN).glob('*.mat')])
    for i, fpath in enumerate(Path(DIR_IN).glob('*.mat')):
        if 'CSR0001A' not in str(fpath):
            continue
        print(f'{fpath.stem}: {i}/{n_files}')
        mpoints = io.loadmat(fpath)['points']
        n_points = mpoints.shape[0]
        if n_points != 6449:
            print(f'error file {face_mat_path.stem}', file=sys.stderr)
            continue

        out_path = f'{DIR_OUT}{fpath.stem}.obj'
        export_mesh(out_path, mpoints, faces, add_one=False)

        ld_points = extract_landmark_coords(mpoints, ld_idxs)
        out_path = f'{DIR_OUT}{fpath.stem}_ld.obj'
        export_vertices(out_path, ld_points)

if __name__  == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    ap.add_argument("-o", "--output", required=True, help="input meta data file")
    ap.add_argument("-f", "--faces", required=True, help="topology")
    ap.add_argument("-l", "--landmarks", required=True, help="landmarks")
    args = vars(ap.parse_args())
    IN_DIR  = args['input']
    OUT_DIR = args['output']
    face_path = args['faces']
    l_path = args['landmarks']

    #test = io.loadmat('/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar/evectors.mat')
    #exit()
    os.makedirs(OUT_DIR, exist_ok=True)

    ld_idxs = io.loadmat(l_path)['landmarksIdxs']
    ld_idxs = ld_idxs.astype(np.int32)
    #for fpath in Path(OUT_DIR).glob('*.*'):
    #    os.remove(fpath)
    convert_mat_to_obj(IN_DIR, OUT_DIR, face_path, ld_idxs)

