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

def export_principal_components(args, pca_model, n_samples):
    for i in range(n_samples):
        var = pca_model.explained_variance_
        param = np.zeros(var.shape)
        param[i] = 3.0*np.sqrt(var[i])
        verts = pca_model.inverse_transform(param)
        n = verts.shape[0]
        verts = verts.reshape(n//3,3)
        opath = join(*[args.out_test_dir, f'{i}st_pca_+3*standard_deviation.obj'])
        export_mesh(fpath=opath, verts=verts, faces=tpl_faces)

        var = pca_model.explained_variance_
        param = np.zeros(var.shape)
        param[i] = -3.0*np.sqrt(var[i])
        verts = pca_model.inverse_transform(param)
        n = verts.shape[0]
        verts = verts.reshape(n//3,3)
        opath = join(*[args.out_test_dir, f'{i}st_pca_-3*standard_deviation.obj'])
        export_mesh(fpath=opath, verts=verts, faces=tpl_faces)

def synthesize(pca_model):
    # #cov_mat = np.diag(np.sqrt(pca_model.singular_values_))
    cov_mat = np.diag(3.0*pca_model.explained_variance_)
    params = np.random.multivariate_normal(np.zeros(shape=pca_model.n_components), cov=cov_mat, size=args.n_samples)
    output_idxs = np.random.randint(low=0, high=args.n_samples, size=args.n_samples)
    for i in tqdm(range(args.n_samples)):
        p = params[i,:]
        if output_idxs[i] < args.n_test_output:
            verts = pca_model.inverse_transform(p)
            n = verts.shape[0]
            verts = verts.reshape(n//3,3)
            opath = join(*[args.out_test_dir, f'{i}.obj'])
            export_mesh(fpath=opath, verts=verts, faces=tpl_faces)

def gen_syn_pca_params(model_path, out_dir, n_samples):
    pca_model = joblib.load(filename=model_path)
    std_range = 2.0
    cov_mat = np.diag(std_range*pca_model.explained_variance_)
    params = np.random.multivariate_normal(np.zeros(shape=pca_model.n_components), cov=cov_mat, size=n_samples)
    for i in range(params.shape[0]):
        p = params[i,:]
        np.save(arr=p, file=os.path.join(*[out_dir, f'syn_{i}.npy']))

def convert_org_mesh_to_pca(model_path, out_dir, vert_paths, scale_factor):
    pca_model = joblib.load(filename=model_path)
    with tqdm(total=len(vert_paths)) as bar:
        for _, path in enumerate(vert_paths):
            with open(str(path), 'rb') as file:
                verts = pickle.load(file)
                verts = (verts * scale_factor).flatten()
                pca_co = pca_model.transform(np.expand_dims(verts, axis=0))
                pca_co = pca_co[0]
                np.save(file=join(*[out_dir, f'{path.stem}.npy']), arr=pca_co)

                bar.update(1)
                bar.set_postfix(msg = f'{pca_co.min()}, {pca_co.max()}')

def male_vert_paths(vert_dir, female_names_dir):
    female_names = set([path.stem for path in Path(female_names_dir).glob('*.*')])
    return [path for path in Path(vert_dir).glob('*.*') if path.stem not in female_names]

def female_vert_paths(vert_dir, female_names_dir):
    female_names = set([path.stem for path in Path(female_names_dir).glob('*.*')])
    return [path for path in Path(vert_dir).glob('*.*') if path.stem in female_names]

def tool_syn_export_pca():
    vdir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/victoria_caesar'
    female_names_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_obj_female/'

    vic_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj'

    #TODO; need to be the same scale factor used while training pca model. check the code pca_vic_tran for this factor
    scale_factor = 0.001

    model_path = '/home/khanhhh//data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_model_vic_male.jlb'
    out_dir = '/home/khanhhh//data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_coords/male/'
    debug_dir = '/home/khanhhh//data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_coords/debug_male/'
    os.makedirs(out_dir, exist_ok=True)
    vert_paths = male_vert_paths(vert_dir=vdir, female_names_dir=female_names_dir)
    convert_org_mesh_to_pca(model_path=model_path, out_dir=out_dir, vert_paths=vert_paths, scale_factor=scale_factor)
    gen_syn_pca_params(model_path=model_path, out_dir=out_dir, n_samples=30000)

    test_syn_pca(model_path=model_path, pca_dir=out_dir, vic_mesh_path=vic_mesh_path, debug_out_dir=debug_dir, n_samples=15)

    model_path = '/home/khanhhh//data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_model_vic_female.jlb'
    out_dir = '/home/khanhhh//data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_coords/female/'
    debug_dir = '/home/khanhhh//data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_coords/debug_female/'
    os.makedirs(out_dir, exist_ok=True)
    vert_paths = female_vert_paths(vert_dir=vdir, female_names_dir=female_names_dir)
    convert_org_mesh_to_pca(model_path=model_path, out_dir=out_dir, vert_paths=vert_paths, scale_factor=scale_factor)
    gen_syn_pca_params(model_path=model_path, out_dir=out_dir, n_samples=30000)

    test_syn_pca(model_path=model_path, pca_dir=out_dir, vic_mesh_path=vic_mesh_path, debug_out_dir=debug_dir, n_samples=15)

def test_syn_pca(model_path, pca_dir, vic_mesh_path, debug_out_dir, n_samples = 15):
    pca_model = joblib.load(filename=model_path)

    co_paths = [path for path in Path(pca_dir).glob('*.*')]
    co_paths = [co_paths[idx] for idx in np.random.randint(0, len(co_paths), n_samples)]

    tpl_verts, tpl_faces = import_mesh(vic_mesh_path)
    NV = tpl_verts.shape[0]

    os.makedirs(debug_out_dir, exist_ok=True)
    for path in Path(debug_out_dir).glob('*.*'):
        os.remove(str(path))

    for path in co_paths:
        p = np.load(path)
        verts = pca_model.inverse_transform(p)
        verts = verts.reshape(NV, 3)
        export_mesh(os.path.join(*[debug_out_dir, f'{path.stem}.obj']), verts=verts, faces=tpl_faces)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-pca_model_path", type=str)
    ap.add_argument("-vic_mesh_path", type=str)
    ap.add_argument("-out_test_dir", type=str)
    ap.add_argument("-out_pca_dir", type=str)
    ap.add_argument("-n_samples", default=30000, required=False, type=int)
    ap.add_argument("-n_test_output", default=200, required=False, type=int)

    args = ap.parse_args()

    os.makedirs(args.out_test_dir, exist_ok=True)
    for path in Path(args.out_test_dir).glob('*.*'):
        os.remove(str(path))

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)

    tool_syn_export_pca()