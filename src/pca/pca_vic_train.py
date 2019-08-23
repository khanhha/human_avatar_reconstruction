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

def export_principal_components(out_dir, pca_model, tpl_faces, n_components, n_std_deviation = 2.0):

    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_components):
        var = pca_model.explained_variance_
        param = np.zeros(var.shape)
        param[i] = n_std_deviation *np.sqrt(var[i])
        verts = pca_model.inverse_transform(param)
        n = verts.shape[0]
        verts = verts.reshape(n//3,3)
        opath = join(*[out_dir, f'pca_component_{i}+{n_std_deviation}*standard_deviation.obj'])
        export_mesh(fpath=opath, verts=verts, faces=tpl_faces)

        var = pca_model.explained_variance_
        param = np.zeros(var.shape)
        param[i] = -n_std_deviation *np.sqrt(var[i])
        verts = pca_model.inverse_transform(param)
        n = verts.shape[0]
        verts = verts.reshape(n//3,3)
        opath = join(*[out_dir, f'pca_component_{i}-{n_std_deviation}*standard_deviation.obj'])
        export_mesh(fpath=opath, verts=verts, faces=tpl_faces)

def load_vertex_array(path):
    out = None
    if path.suffix == '.pkl':
        with open(str(path), 'rb') as file:
            verts = pickle.load(file)
            out = verts.flatten()
    if path.suffix == '.npy':
        verts = np.load(str(path))
        out = verts.flatten()
    elif path.suffix == '.obj':
        verts, _ = import_mesh_obj(str(path))
        out = verts.flatten()
    else:
        assert 'unsupported vertex file format'
    return out

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
            path = paths[i_start+i]
            V_tmp[i, :] = load_vertex_array(path)

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
            verts = load_vertex_array(path)
            pca_co = pca_model.transform(np.expand_dims(verts, axis=0))
            pca_co = pca_co[0]
            np.save(file=join(*[out_dir, f'{path.stem}.npy']), arr=pca_co)

            bar.update(1)
            bar.set_postfix(msg = f'{pca_co.min()}, {pca_co.max()}')

def tool_export_pca_coords(model_path, vert_paths, out_pca_dir, synthesize_samples=0):
    os.makedirs(out_pca_dir, exist_ok=True)
    print(f'\ttransforming {len(vert_paths)} original mesh to pca values')
    convert_org_mesh_to_pca(model_path=model_path, out_dir=out_pca_dir, vert_paths=vert_paths)
    if synthesize_samples > 0:
        print(f'\tstart synthesizing {synthesize_samples} pca values. model path = {Path(model_path).name}')
        gen_syn_pca_params(model_path=model_path, out_dir=out_pca_dir, n_samples=synthesize_samples)

#test_syn_pca(model_path=model_path, pca_dir=out_pca_dir, vic_mesh_path=vic_mesh_path, debug_out_dir=debug_dir, n_samples=15)

def export_random_pca_mesh(model_path, pca_dir, vic_mesh_path, debug_out_dir, n_samples = 15):
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

def tool_transform_pca_to_verts(model_path, pca_dir, vert_dir):
    os.makedirs(vert_dir, exist_ok=True)
    for path in Path(vert_dir).glob("*.*"):
        os.remove(str(path))

    pca_model = joblib.load(filename=model_path)
    co_paths = [path for path in Path(pca_dir).glob('*.*')]

    for path in co_paths:
        p = np.load(path)
        verts = pca_model.inverse_transform(p)
        verts = verts.reshape(-1, 3)
        np.save(file = os.path.join(*[vert_dir, f'{path.stem}.npy']), arr=verts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-vert_dir", type=str, required=True, help="directory that contains the caesar mesh vertices: *.pkl files or *.obj files")
    ap.add_argument("-vic_mesh_path", type=str, required=True, help="path to victoria template mesh file")
    ap.add_argument("-out_dir",  type=str, required=True, help="ouput directory that contains everything output data")
    ap.add_argument("-pca_k", type=int, required=False, default=50, help="the number of PCA components to keep. stick to 50 for now")
    ap.add_argument("-female_names_file", type=str, required=False, help="path to txt file that contains caesar female names. this this file is provided, separate models for famale and female will be trained")
    ap.add_argument("-n_synthesize_samples", type=int, required=False, default=0, help="the number of meshes to synthesize. if it is zero. synthesize no meshes")
    ap.add_argument("-n_debug_samples", type=int, required=False, default=50, help="number of synthesized PCA values to export to mesh")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vpaths = [path for path in Path(args.vert_dir).glob('*.*')]
    extensions = set([path.suffix for path in vpaths])
    assert len(extensions) ==1, f'vertex folder contains more than one file type: {extensions}'

    male_paths = None
    female_paths = None
    if args.female_names_file is not None:
        female_names = set()
        with open(args.female_names_file, 'r') as file:
            for line in file.readlines():
                name = line.replace('\n','')
                assert '.obj' not in name, 'incorrect name format'
                female_names.add(name)
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
        #########################################################################
        print(f'train male pca model incrementally: n files = {len(male_paths)}')
        male_model   = train_pca(male_paths, NV=NV)
        opath_male_model = os.path.join(*[args.out_dir, 'vic_male_pca_model.jlb'])
        joblib.dump(male_model, filename=opath_male_model)
        print(f'dump male model to file {opath_male_model}')

        test_pca_comp_dir = os.path.join(*[args.out_dir, 'debug_pca_components_male'])
        print(f'export male principal components to {test_pca_comp_dir}')
        export_principal_components(out_dir=test_pca_comp_dir, pca_model=male_model, tpl_faces=tpl_faces, n_components=10, n_std_deviation=3.0)

        male_pca_co_dir = os.path.join(*[args.out_dir, 'pca_coords', 'male'])
        print(f'debug: exporting male pca coordinates to {male_pca_co_dir}')
        tool_export_pca_coords(opath_male_model, male_paths, male_pca_co_dir, synthesize_samples=args.n_synthesize_samples)

        male_vert_dir = os.path.join(*[args.out_dir, 'verts', 'male'])
        print(f'transforming male pca coordinates to vertex array (for projection to front/side silhouette)')
        tool_transform_pca_to_verts(model_path=opath_male_model, pca_dir= male_pca_co_dir, vert_dir= male_vert_dir)

        if args.n_debug_samples > 0:
            debug_mesh_male_dir =  os.path.join(*[args.out_dir, 'debug_male_syned_mesh'])
            print(f'exporting {args.n_debug_samples} random male pca meshes to {debug_mesh_male_dir}')
            export_random_pca_mesh(model_path=opath_male_model, pca_dir=male_pca_co_dir, vic_mesh_path=args.vic_mesh_path, debug_out_dir=debug_mesh_male_dir, n_samples=args.n_debug_samples)

        #########################################################################
        print(f'\n\ntrain female pca model incrementally: n files = {len(female_paths)}')
        female_model = train_pca(female_paths, NV=NV)
        opath_female_model = os.path.join(*[args.out_dir, 'vic_female_pca_model.jlb'])
        joblib.dump(female_model, filename=opath_female_model)
        print(f'dump female model to file {opath_female_model}')

        test_pca_comp_dir = os.path.join(*[args.out_dir, 'debug_pca_components_female'])
        print(f'export male principal componets to {test_pca_comp_dir}')
        export_principal_components(out_dir=test_pca_comp_dir, pca_model=female_model, tpl_faces=tpl_faces, n_components=10, n_std_deviation=3.0)

        female_pca_co_dir = os.path.join(*[args.out_dir, 'pca_coords', 'female'])
        print(f'debug: exporting female pca coordinates to {female_pca_co_dir}')
        tool_export_pca_coords(opath_female_model, female_paths, female_pca_co_dir, synthesize_samples=args.n_synthesize_samples)

        female_vert_dir = os.path.join(*[args.out_dir, 'verts', 'female'])
        print(f'transforming male pca coordinates to vertex array (for projection to front/side silhouette)')
        tool_transform_pca_to_verts(model_path=opath_female_model, pca_dir= female_pca_co_dir, vert_dir= female_vert_dir)

        if args.n_debug_samples > 0:
            debug_mesh_female_dir =  os.path.join(*[args.out_dir, 'debug_female_syned_mesh'])
            print(f'exporting {args.n_debug_samples} random female pca meshes to {debug_mesh_female_dir}')
            export_random_pca_mesh(model_path=opath_female_model, pca_dir=female_pca_co_dir, vic_mesh_path=args.vic_mesh_path, debug_out_dir=debug_mesh_female_dir, n_samples=args.n_debug_samples)

if __name__ == '__main__':
    main()