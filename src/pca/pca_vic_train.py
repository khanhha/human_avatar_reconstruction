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

def train_pca(paths, v_scale_factor, NV):

    step_size = 100

    pca_model = IncrementalPCA(n_components=50, whiten=False)
    N = len(paths)
    bar = tqdm(total=N)

    for i_start in range(0, N, step_size):
        i_end = min(i_start+step_size, N)
        if i_end - i_start < pca_model.n_components:
            print(f'ignore the batch {i_start} of size {i_end-i_start}. < n_components = {pca_model.n_components}')
            continue

        bar.update(i_end-i_start)
        bar.set_postfix(range=f"{i_start} : {i_end}. len={i_end-i_start}")
        V_tmp = np.zeros((i_end-i_start, NV*3), dtype=np.float)
        for i in range(i_end-i_start):
            with open(str(paths[i_start+i]), 'rb') as file:
                verts = pickle.load(file)
                verts *= v_scale_factor
                V_tmp[i,:] = verts.flatten()
                del verts
        pca_model.partial_fit(V_tmp)
        del V_tmp
        gc.collect()

    bar.close()

    return pca_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-vert_dir", type=str, required=True, help="")
    ap.add_argument("-vic_mesh_path", type=str, required=True, help="")
    ap.add_argument("-out_dir",  type=str, required=True, help="")
    ap.add_argument("-pca_k", type=int, required=False, default=50)
    ap.add_argument("-scale_factor", type=float, required=False, default=0.001)
    ap.add_argument("-female_names_dir", type=str, required=False, default=None)

    args = ap.parse_args()

    vpaths = [path for path in Path(args.vert_dir).glob('*.pkl')]
    male_paths = None
    female_paths = None
    if args.female_names_dir is not None:
        female_names = set([path.stem for path in Path(args.female_names_dir).glob('*.*')])
        male_paths = [path for path in vpaths if path.stem not in female_names]
        female_paths = [path for path in vpaths if path.stem in female_names]

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)
    NV = tpl_verts.shape[0]
    vsize = tpl_verts.size
    assert vsize == (tpl_verts.shape[0] * tpl_verts.shape[1])

    if male_paths is None or female_paths is None:
        print(f'train joint pca model: n files = {len(vpaths)}')
        pca_model = train_pca(vpaths, v_scale_factor=args.scale_factorm, NV=NV)
        joblib.dump(pca_model, filename=os.path.join(*[args.out_dir, 'vic_joint_pca_model.jlb']))
    else:
        print(f'train male pca model: n files = {len(male_paths)}')
        male_model   = train_pca(male_paths,   v_scale_factor=args.scale_factor, NV=NV)
        opath = os.path.join(*[args.out_dir, 'vic_male_pca_model.jlb'])
        joblib.dump(male_model, filename=opath)
        print(f'dump male model to file {opath}')

        print(f'train female pca model: n files = {len(female_paths)}')
        female_model = train_pca(female_paths, v_scale_factor=args.scale_factor, NV=NV)
        opath = os.path.join(*[args.out_dir, 'vic_female_pca_model.jlb'])
        joblib.dump(female_model, filename=opath)
        print(f'dump female model to file {opath}')

    # if args.out_pca_co_dir != '':
    #     bar = tqdm(total=len(vpaths))
    #     for _, path in enumerate(vpaths):
    #         with open(str(path), 'rb') as file:
    #             verts = pickle.load(file)
    #             verts = (verts * args.scale_factor).flatten()
    #             pca_co = pca_model.transform(np.expand_dims(verts, axis=0))
    #             np.save(file=join(*[args.out_pca_co_dir, f'{path.stem}.npy']), arr=pca_co[0])
    #
    #             bar.update(1)
    #             bar.set_postfix(msg = f' pca min, max: {pca_co.min()}, {pca_co.max()}')
    #     bar.close()

if __name__ == '__main__':
    main()