import pickle
import argparse
from pathlib import Path
import os
from tqdm import tqdm
from common.obj_util import import_mesh, export_mesh
import numpy as np
from os.path import join

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-in_dir", type=str, required=True, help="")
    ap.add_argument("-map_path", type=str, required=True, help="")
    ap.add_argument("-alice_mesh_path", type=str, required=True, help="")
    ap.add_argument("-out_dir", type=str, required=True, help="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in Path(args.out_dir).glob('*.obj'):
        os.remove(str(path))

    with open(args.map_path, 'rb') as file:
        all_v_maps = pickle.load(file)

    alc_verts, alc_faces = import_mesh(args.alice_mesh_path)
    n_alc_verts = len(alc_verts)

    paths = [path for path in Path(args.in_dir).glob('*.obj')]
    for i, path in tqdm(enumerate(paths)):
        verts, faces = import_mesh(path)

        alc_caesar_verts = np.zeros((n_alc_verts, 3), dtype=np.float)

        for alc_idx in range(n_alc_verts):
            v_maps = all_v_maps[alc_idx]
            v_idxs = [pair[0] for pair in v_maps]
            v_weights = [pair[1] for pair in v_maps]
            avg_co = np.average(verts[v_idxs, :], axis=0, weights=v_weights)
            alc_caesar_verts[alc_idx, :] = avg_co

        export_mesh(join(*[args.out_dir, path.name]), verts=alc_caesar_verts, faces=alc_faces)

        if i > 20:
            break