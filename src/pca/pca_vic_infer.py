import scipy.io as io
import argparse
import os
from os.path import join
import numpy as np
from pathlib import Path
import pickle
from common.obj_util import export_mesh, import_mesh
from pca.pca_util import load_faces
from tqdm import tqdm
import gc
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-vert_dir", type=str, required=True, help="")
    ap.add_argument("-vic_mesh_path", type=str, required=True, help="")
    ap.add_argument("-pca_path",  type=str, required=True, help="")
    ap.add_argument("-out_dir",  type=str, required=True, help="")
    args = ap.parse_args()

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)
    vsize = tpl_verts.size
    NV = tpl_verts.shape[0]
    assert vsize == (tpl_verts.shape[0] * tpl_verts.shape[1])

    pca_model = joblib.load(filename=args.pca_path)

    for path in Path(args.out_dir).glob('*.obj'):
        os.remove(str(path))

    vpaths = [path for path in Path(args.vert_dir).glob('*.pkl')]
    n_tries = 5
    for _ in tqdm(range(0, n_tries)):
        path = vpaths[np.random.randint(0, len(vpaths))]
        with open(str(path), 'rb') as file:
            verts = pickle.load(file)
            verts = verts*0.001

            pca_co = pca_model.transform(np.expand_dims(verts.flatten(), axis=0))

            verts_1 = pca_model.inverse_transform(pca_co)[0]
            verts_1 = verts_1.reshape((NV, 3))

            export_mesh(fpath=join(*[args.out_dir, f'{path.stem}.obj']), verts=verts_1, faces=tpl_faces)
            export_mesh(fpath=join(*[args.out_dir, f'{path.stem}_truth.obj']), verts=verts, faces=tpl_faces)

if __name__ == '__main__':
    main()
