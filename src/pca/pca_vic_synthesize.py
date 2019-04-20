import scipy.io as io
import argparse
import os
from os.path import join
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from pca.pca_util import load_faces
from common.obj_util import export_mesh, import_mesh
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-pca_model_path", type=str)
    ap.add_argument("-vic_mesh_path", type=str)
    ap.add_argument("-out_dir", type=str)
    ap.add_argument("-n_samples", default=30, required=False, type=int)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in Path(args.out_dir).glob('*.obj'):
        os.remove(str(path))

    n_samples = args.n_samples

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)

    pca_model = joblib.load(filename=args.pca_model_path)

    # #cov_mat = np.diag(np.sqrt(pca_model.singular_values_))
    # cov_mat = np.diag(2.0*pca_model.explained_variance_)
    # params = np.random.multivariate_normal(np.zeros(shape=pca_model.n_components), cov=cov_mat, size=n_samples)
    # for i in range(n_samples):
    #     p = params[i,:]
    #     verts = pca_model.inverse_transform(p)
    #     n = verts.shape[0]
    #     verts = verts.reshape(n//3,3)
    #     opath = join(*[args.out_dir, f'{i}.obj'])
    #     export_mesh(fpath=opath, verts=verts, faces=tpl_faces)

    for i in range(n_samples):
        var = pca_model.explained_variance_
        param = np.zeros(var.shape)
        param[i] = 3.0*np.sqrt(var[i])
        verts = pca_model.inverse_transform(param)
        n = verts.shape[0]
        verts = verts.reshape(n//3,3)
        opath = join(*[args.out_dir, f'{i}st_pca_+3*standard_deviation.obj'])
        export_mesh(fpath=opath, verts=verts, faces=tpl_faces)

        var = pca_model.explained_variance_
        param = np.zeros(var.shape)
        param[i] = -3.0*np.sqrt(var[i])
        verts = pca_model.inverse_transform(param)
        n = verts.shape[0]
        verts = verts.reshape(n//3,3)
        opath = join(*[args.out_dir, f'{i}st_pca_-3*standard_deviation.obj'])
        export_mesh(fpath=opath, verts=verts, faces=tpl_faces)