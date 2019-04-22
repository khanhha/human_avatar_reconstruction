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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-vert_dir", type=str, required=True, help="")
    ap.add_argument("-pca_path",  type=str, required=True, help="")
    ap.add_argument("-out_dir",  type=str, required=True, help="")
    ap.add_argument("-caesar_vic_height", type=str, required=True)
    ap.add_argument("-scale_factor", type=float, required=False, default=0.001)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    vpaths = [path for path in Path(args.vert_dir).glob('*.pkl')]
    # bar = tqdm(total=len(vpaths))
    # heights = {}
    # for idx, path in enumerate(vpaths):
    #     with open(str(path), 'rb') as file:
    #         verts = pickle.load(file)
    #         verts = verts*args.scale_factor
    #         heights[path.stem] = verts[:,2].max() - verts[:,2].min()
    #     bar.update(1)
    # bar.close()
    # joblib.dump(value=heights, filename=args.caesar_vic_height)

    pca_model = joblib.load(filename=args.pca_path)
    bar = tqdm(total=len(vpaths))
    all_pca_cos = []
    for _, path in enumerate(vpaths):
        with open(str(path), 'rb') as file:
            verts = pickle.load(file)
            verts = (verts*args.scale_factor).flatten()
            pca_co = pca_model.transform(np.expand_dims(verts, axis=0))
            pca_co = pca_co[0]
            np.save(file=join(*[args.out_dir, f'{path.stem}.npy']), arr=pca_co)

            all_pca_cos.append(pca_co)

            bar.update(1)
            bar.set_postfix(msg = f'{pca_co.min()}, {pca_co.max()}')

    all_pca_cos  = np.array(all_pca_cos)

    print(f'min, max before scaling: {all_pca_cos.min()}, {all_pca_cos.max()}')
    target_trans = MinMaxScaler()
    target_trans.fit(all_pca_cos)
    transformed_pca_cos = target_trans.transform(all_pca_cos)
    print(f'min, max after scaling: {transformed_pca_cos.min()}, {transformed_pca_cos.max()}')

if __name__ == '__main__':
    main()