import pickle
import argparse
from pathlib import Path
import os
from tqdm import tqdm
from common.obj_util import import_mesh_obj, export_mesh
import numpy as np
from os.path import join
import multiprocessing
from deformation.ffdt_deformation_lib import TemplateMeshDeform
from random import shuffle
import gc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-in_dir", type=str, required=True, help="")
    ap.add_argument("-template_mesh_path", type=str, required=True, help="")
    ap.add_argument("-control_mesh_path", type=str, required=True, help="")
    ap.add_argument("-parameterization_path", type=str, required=True, help="")
    ap.add_argument("-out_dir", type=str, required=True, help="")
    ap.add_argument("-n_process", type=int, required=False, default=4, help="")
    ap.add_argument("-ouput_debug", type=int, required=False, default=1, help="")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    #cur_paths = [path for path in Path(args.out_dir).glob('*.obj')]
    cur_paths = []

    tpl_verts, tpl_faces = import_mesh_obj(args.template_mesh_path)
    n_tpl_verts = len(tpl_verts)

    ctl_verts, ctl_faces = import_mesh_obj(args.control_mesh_path)

    with open(args.parameterization_path, 'rb') as f:
        data = pickle.load(f)
        vert_UVWs = data['template_vert_UVW']
        vert_weights = data['template_vert_weight']
        vert_effect_idxs = data['template_vert_effect_idxs']

        deform = TemplateMeshDeform(effective_range=4, use_mean_rad=False)
        deform.set_meshes(ctl_verts=ctl_verts, ctl_tris=ctl_faces, tpl_verts=tpl_verts, tpl_faces=tpl_faces)
        deform.set_parameterization(vert_tri_UVWs=vert_UVWs, vert_tri_weights=vert_weights, vert_effect_tri_idxs=vert_effect_idxs)

        del vert_UVWs
        del vert_weights
        del vert_effect_idxs
        del data

    gc.collect()

    paths = [path for path in Path(args.in_dir).glob('*.obj')]
    cur_paths_set = set([path.name for path in cur_paths])
    paths = [path for path in paths if path.name not in cur_paths_set]
    shuffle(paths)

    #'CSR1428A'
    # for path in paths:
    #     if 'CSR1428A' not in path.name:
    #         continue
    #     print(path)
    #     ctl_df_verts, ctl_df_faces = import_mesh(fpath=path)
    #
    #     tpl_new_verts, _ = deform.deform(ctl_df_verts)
    #
    #     out_path = join(*[args.out_dir, path.name])
    #     export_mesh(fpath=out_path, verts=tpl_new_verts, faces=tpl_faces)
    #
    # exit()

    #paths = paths[:12]
    n_files = len(paths)
    def parallel_util(paths, start, end):
        with tqdm(total = end-start) as pbar:
            for i, path in enumerate(paths[start:end]):
                ctl_df_verts, ctl_df_faces = import_mesh_obj(fpath=path)
                # this scale is for MPII
                #ctl_df_verts *= 0.01 #rescale to the same approximation of the paramiterization to increase accuracy

                # this scale is for UCSC
                #ctl_df_verts *= 10.0
                tpl_new_verts = deform.deform(ctl_df_verts)

                #tpl_new_verts *= 0.1 #scale the final mesh down, so that all the meshes have the height in meter: 1.7 for example

                out_path = join(*[args.out_dir, f'{path.stem}.pkl'])
                with open(out_path, 'wb') as file:
                    pickle.dump(obj=tpl_new_verts, file=file, protocol=pickle.HIGHEST_PROTOCOL)

                if np.random.rand() < 0.1:
                    out_path = join(*[args.out_dir, f'{path.stem}.obj'])
                    export_mesh(fpath=out_path, verts=tpl_new_verts, faces=tpl_faces)

                pbar.update(1)

    N = len(paths)
    step = N//args.n_process
    procs = []
    for i in range(args.n_process):
        start = i*step
        end = (i+1)*step
        if i == args.n_process-1 and end != N:
            end = N
        p = multiprocessing.Process(target=parallel_util, args=(paths, start, end))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()