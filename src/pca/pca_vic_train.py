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
    ap.add_argument("-tmp_data_path", type=str, required=True, help="")
    ap.add_argument("-out_pca_path",  type=str, required=True, help="")
    ap.add_argument("-pca_k", type=int, required=False, default=50)
    ap.add_argument("-scale_factor", type=float, required=False, default=0.001)

    args = ap.parse_args()

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)

    vsize = tpl_verts.size
    NV = tpl_verts.shape[0]
    assert vsize == (tpl_verts.shape[0] * tpl_verts.shape[1])

    vpaths = [path for path in Path(args.vert_dir).glob('*.pkl')]
    N = len(vpaths)
    step_size = 100

    pca_model = IncrementalPCA(n_components=50, whiten=False)

    bar = tqdm(total=N)
    for i_start in range(0, N, step_size):
        i_end = min(i_start+step_size, N)
        if i_end  - i_start < step_size:
            print(f'ignore the last batch: size =  {i_end-i_start}')
            break
        bar.update(i_end-i_start)
        bar.set_postfix(range=f"{i_start} : {i_end}. len={i_end-i_start}")
        V_tmp = np.zeros((i_end-i_start, NV * 3), dtype=np.float)
        for i in range(i_end-i_start):
            with open(str(vpaths[i_start+i]), 'rb') as file:
                verts = pickle.load(file)
                verts *= args.scale_factor
                V_tmp[i,:] = verts.flatten()
                del verts

        pca_model.partial_fit(V_tmp)
        del V_tmp
        gc.collect()

    bar.close()

    joblib.dump(pca_model, filename=args.out_pca_path)

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