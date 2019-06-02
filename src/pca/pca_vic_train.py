import scipy.io as io
import argparse
import os
from os.path import join
import numpy as np
from pathlib import Path
import pickle
from common.obj_util import export_mesh, import_mesh_obj
from pca.pca_util import load_faces
from tqdm import tqdm
import gc
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

def train_pca(paths, NV):

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
                V_tmp[i,:] = verts.flatten()
                del verts
        pca_model.partial_fit(V_tmp)
        del V_tmp
        gc.collect()

    bar.close()

    return pca_model

def gen_syn_pca_params(model_path, out_dir, n_samples):
    pca_model = joblib.load(filename=model_path)
    std_range = 2.0
    cov_mat = np.diag(std_range*pca_model.explained_variance_)
    params = np.random.multivariate_normal(np.zeros(shape=pca_model.n_components), cov=cov_mat, size=n_samples)
    for i in range(params.shape[0]):
        p = params[i,:]
        np.save(arr=p, file=os.path.join(*[out_dir, f'syn_{i}.npy']))

def convert_org_mesh_to_pca(model_path, out_dir, vert_paths):
    pca_model = joblib.load(filename=model_path)
    with tqdm(total=len(vert_paths)) as bar:
        for _, path in enumerate(vert_paths):
            with open(str(path), 'rb') as file:
                verts = pickle.load(file)
                verts = verts.flatten()
                pca_co = pca_model.transform(np.expand_dims(verts, axis=0))
                pca_co = pca_co[0]
                np.save(file=join(*[out_dir, f'{path.stem}.npy']), arr=pca_co)

                bar.update(1)
                bar.set_postfix(msg = f'{pca_co.min()}, {pca_co.max()}')

def male_vert_paths(vert_dir, female_names_dir):
    female_names = set([path.stem for path in Path(female_names_dir).glob('*.*')])
    return [path for path in Path(vert_dir).glob('*.pkl') if path.stem not in female_names]

def female_vert_paths(vert_dir, female_names_dir):
    female_names = set([path.stem for path in Path(female_names_dir).glob('*.pkl')])
    return [path for path in Path(vert_dir).glob('*.pkl') if path.stem in female_names]

def tool_syn_export_pca(args, model_dir):
    vdir = args.vert_dir
    female_names_dir = args.female_names_dir

    vic_mesh_path = args.vic_mesh_path

    model_path =   os.path.join(*[model_dir,   'vic_male_pca_model.jlb'])
    out_dir =   os.path.join(*[model_dir,   'pca_coords', 'male'])
    debug_dir =   os.path.join(*[model_dir, 'pca_coords', 'debug_male'])
    os.makedirs(out_dir, exist_ok=True)
    vert_paths = male_vert_paths(vert_dir=vdir, female_names_dir=female_names_dir)
    convert_org_mesh_to_pca(model_path=model_path, out_dir=out_dir, vert_paths=vert_paths)
    gen_syn_pca_params(model_path=model_path, out_dir=out_dir, n_samples=30000)

    test_syn_pca(model_path=model_path, pca_dir=out_dir, vic_mesh_path=vic_mesh_path, debug_out_dir=debug_dir, n_samples=15)

    model_path = os.path.join(*[model_dir, 'vic_female_pca_model.jlb'])
    out_dir =   os.path.join(*[model_dir, 'pca_coords', 'female'])
    debug_dir = os.path.join(*[model_dir, 'pca_coords', 'debug_female'])
    os.makedirs(out_dir, exist_ok=True)
    vert_paths = female_vert_paths(vert_dir=vdir, female_names_dir=female_names_dir)
    convert_org_mesh_to_pca(model_path=model_path, out_dir=out_dir, vert_paths=vert_paths)
    gen_syn_pca_params(model_path=model_path, out_dir=out_dir, n_samples=30000)

    test_syn_pca(model_path=model_path, pca_dir=out_dir, vic_mesh_path=vic_mesh_path, debug_out_dir=debug_dir, n_samples=15)

def test_syn_pca(model_path, pca_dir, vic_mesh_path, debug_out_dir, n_samples = 15):
    pca_model = joblib.load(filename=model_path)

    co_paths = [path for path in Path(pca_dir).glob('*.*')]
    co_paths = [co_paths[idx] for idx in np.random.randint(0, len(co_paths), n_samples)]

    tpl_verts, tpl_faces = import_mesh_obj(vic_mesh_path)
    NV = tpl_verts.shape[0]

    os.makedirs(debug_out_dir, exist_ok=True)
    for path in Path(debug_out_dir).glob('*.*'):
        os.remove(str(path))

    for path in co_paths:
        p = np.load(path)
        verts = pca_model.inverse_transform(p)
        verts = verts.reshape(NV, 3)
        export_mesh(os.path.join(*[debug_out_dir, f'{path.stem}.obj']), verts=verts, faces=tpl_faces)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-vert_dir", type=str, required=True, help="")
    ap.add_argument("-vic_mesh_path", type=str, required=True, help="")
    ap.add_argument("-out_dir",  type=str, required=True, help="")
    ap.add_argument("-pca_k", type=int, required=False, default=50)
    ap.add_argument("-female_names_dir", type=str, required=False, default=None)
    ap.add_argument("-do_synthesize", type=int, required=False, default=0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vpaths = [path for path in Path(args.vert_dir).glob('*.pkl')]
    male_paths = None
    female_paths = None
    if args.female_names_dir is not None:
        female_names = set([path.stem for path in Path(args.female_names_dir).glob('*.*')])
        male_paths = [path for path in vpaths if path.stem not in female_names]
        female_paths = [path for path in vpaths if path.stem in female_names]

    tpl_verts, tpl_faces = import_mesh_obj(args.vic_mesh_path)
    NV = tpl_verts.shape[0]
    vsize = tpl_verts.size
    assert vsize == (tpl_verts.shape[0] * tpl_verts.shape[1])

    if male_paths is None or female_paths is None:
        print(f'train joint pca model: n files = {len(vpaths)}')
        pca_model = train_pca(vpaths, NV=NV)
        joblib.dump(pca_model, filename=os.path.join(*[args.out_dir, 'vic_joint_pca_model.jlb']))
    else:
        # print(f'train male pca model: n files = {len(male_paths)}')
        # male_model   = train_pca(male_paths, NV=NV)
        # opath = os.path.join(*[args.out_dir, 'vic_male_pca_model.jlb'])
        # joblib.dump(male_model, filename=opath)
        # print(f'dump male model to file {opath}')
        #
        # print(f'train female pca model: n files = {len(female_paths)}')
        # female_model = train_pca(female_paths, NV=NV)
        # opath = os.path.join(*[args.out_dir, 'vic_female_pca_model.jlb'])
        # joblib.dump(female_model, filename=opath)
        # print(f'dump female model to file {opath}')

        if args.do_synthesize == 1:
            tool_syn_export_pca(args, args.out_dir)

if __name__ == '__main__':
    main()