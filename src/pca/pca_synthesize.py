import scipy.io as io
import argparse
import os
from os.path import join
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from pca.pca_util import load_faces
from common.obj_util import export_mesh

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir", type=str)
    ap.add_argument("-out_dir", type=str)
    ap.add_argument("-n_pca", default=50, required=False, type=int)
    ap.add_argument("-n_samples", default=200, required=False, type=int)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in Path(args.out_dir).glob('*.obj'):
        os.remove(str(path))

    n_vert = 6449
    n_face = 12894

    n_samples = args.n_samples

    faces = load_faces(f'{args.model_dir}/model.dat')
    mean_points  = io.loadmat(f'{args.model_dir}/meanShape.mat')['points'] #shape=(6449,3)
    evectors_org = io.loadmat(join(*[args.model_dir, 'evectors.mat']))['evectors']
    evalues_org  = io.loadmat(join(*[args.model_dir, 'evalues.mat']))['evalues']

    npca = args.n_pca
    evectors = evectors_org[:npca, :].T
    evalues  = evalues_org[:,:npca][0]
    sdvalues = np.sqrt(evalues)

    cov_mat = np.diag(evalues)
    params = np.random.multivariate_normal(np.zeros(shape=(npca)), cov=cov_mat, size=n_samples)

    for i in tqdm(range(n_samples)):
        nparams = params[i,:]
        verts = np.dot(evectors, nparams.reshape(npca, 1))
        verts = verts.reshape(3, n_vert) + mean_points.T
        verts *= 0.01
        opath = join(*[args.out_dir, f'synthesized_{i}.obj'])
        #print(f'export file {opath}')
        export_mesh(opath, verts=verts.T, faces=faces)