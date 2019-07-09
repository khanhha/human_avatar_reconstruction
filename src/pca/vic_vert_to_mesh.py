import argparse
from pathlib import Path
import pickle
import os
from common.obj_util import  import_mesh_obj, export_mesh

#how to run the file
# cd the folder src
# export PYTHONPATH="${PYTHONPATH}:./"
# python ./pca/vic_vert_to_mesh.py -tpl_mesh_path path_to_vic_mesh.obj -vert_dir dir_to_vic_caesar_verts -out_dir OUT_DIR

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tpl_mesh_path", type=str, required=True, help="victoria template mesh path")
    ap.add_argument("-vert_dir", type=str, required=True, help="directory contains pkl files")
    ap.add_argument("-out_dir", type=str, required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'output obj files to :', args.out_dir)

    tpl_verts, tpl_faces = import_mesh_obj(args.tpl_mesh_path)
    paths = [path for path in Path(args.vert_dir).glob("*.pkl")]

    cnt = 0
    for path in paths[:100]:
        if cnt % 100 == 0:
            print(f'processed: {cnt}/{len(paths)}')

        with open(str(path), 'rb') as file:
            verts = pickle.load(file)
            opath = os.path.join(*[args.out_dir, f'{path.stem}.obj'])
            export_mesh(opath, verts, tpl_faces)

        cnt = cnt + 1
