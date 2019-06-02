import pickle
import argparse
from pathlib import Path
import os
from tqdm import tqdm
from common.obj_util import import_mesh_obj, export_mesh
import numpy as np
from os.path import join
import multiprocessing
from deformation.ffdt_deformation_lib import TemplateMeshDeform
from random import shuffle
import gc
from functools import partial

def export(out_dir, faces, path):
    with open(str(path), 'rb') as file:
        verts = pickle.load(file)
    opath = join(*[args.out_dir, f'{path.stem}.obj'])
    export_mesh(fpath=str(opath), verts=verts, faces=tpl_faces)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vert_dir", type=str, required=True, help="")
    ap.add_argument("-vic_mesh_path", type=str, required=True, help="")
    ap.add_argument("-out_dir", type=str, required=True, help="")
    args = ap.parse_args()

    _, tpl_faces = import_mesh_obj(args.vic_mesh_path)

    for path in Path(args.out_dir).glob('*.obj'):
        os.remove(str(path))

    paths = [path for path in Path(args.vert_dir).glob('*.pkl')]

    pool = multiprocessing.Pool(12)
    for _ in tqdm(pool.imap_unordered(partial(export, args.out_dir, tpl_faces), paths)):
        pass
    pool.join()
    pool.close()

    # for idx, v_path in tqdm(enumerate()):
    #     with open(str(v_path), 'rb') as file:
    #         verts = pickle.load(file)
    #     opath = join(*[args.out_dir, f'{v_path.stem}.obj'])
    #     export_mesh(fpath=str(opath), verts=verts, faces=tpl_faces)