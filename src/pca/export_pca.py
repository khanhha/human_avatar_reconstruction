import scipy.io as io
import argparse
import os
from os.path import join
import numpy as np
from pathlib import Path

from common.obj_util import export_mesh, import_mesh

def load_faces(path):
   with open(path, 'r') as file:
       lines = file.readlines()
       nvert = int(lines[0].split(' ')[0])
       nface = int(lines[0].split(' ')[1])
       face_lines = lines[nvert+1:nvert+1+nface]
       faces = []
       for line in face_lines:
           fv_strs = line.split(' ')
           assert  len(fv_strs) == 4
           fv_strs = fv_strs[:3]
           fv_idxs = [int(vstr) for vstr in fv_strs]
           faces.append(fv_idxs)

       assert len(faces) == 12894

       return faces

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-in_dir", type=str)
    ap.add_argument("-mesh_dir", type=str)
    args = ap.parse_args()


    n_vert = 6449
    n_face = 12894
    faces = load_faces(f'{args.in_dir}/model.dat')
    mean_points = io.loadmat(f'{args.in_dir}/meanShape.mat')['points'] #shape=(6449,3)
    #export_mesh(os.path.join(*[args.in_dir, 'mean_mesh.obj']), verts=mean_points, faces=faces)
    evectors_org = io.loadmat(join(*[args.in_dir, 'evectors.mat']))['evectors']
    evalues_org  = io.loadmat(join(*[args.in_dir, 'evalues.mat']))['evalues']
    print(evectors_org.shape)
    print(evalues_org.shape)

    npca = 200
    evectors = evectors_org[:npca, :].T
    evalues  = evalues_org[:,:npca]


    #out_dir = join(*[args.in_dir, 'test_reconstruct'])
    #os.makedirs(out_dir, exist_ok=True)

    # mesh_paths = [path for path in Path(args.mesh_dir).glob('*.obj')]
    # for i in range(1):
    #     mpath = mesh_paths[np.random.randint(0, len(mesh_paths))]
    #     mverts_org, _ = import_mesh(mpath)
    #
    #     diff  = mverts_org.T - mean_points.T
    #     diff  = diff.flatten()
    #
    #     params = np.zeros(npca, dtype=np.float)
    #     for k in range(npca):
    #         params[k] = np.dot(diff, evectors[:, k])
    #
    #     verts_1 = np.dot(evectors, params.reshape(npca, 1))
    #     verts_1 = verts_1.reshape(3, n_vert) + mean_points.T
    #     verts_1 = verts_1.T
    #
    #     opath = join(*[out_dir, f'evector_{i}_org.obj'])
    #     export_mesh(opath, mverts_org, faces)
    #     opath = join(*[out_dir, f'evector_{i}.obj'])
    #     export_mesh(opath, verts_1, faces)


    out_dir = join(*[args.in_dir, 'evector_samples'])
    os.makedirs(out_dir, exist_ok=True)
    sdvalues = np.sqrt(evalues)
    for i in range(30):
        #verts = evectors[i, :].reshape(n_vert, 3)
        nparams = np.zeros(npca, dtype=np.float)
        for j in range(npca):
            p = np.random.uniform(-3*sdvalues[0,j], 3*sdvalues[0,j], 1)
            nparams[j] = p[0]

        #nparams = np.zeros(npca, dtype=np.float)
        #nparams[i] = (3*np.sqrt(evalues[0, i]))
        verts = np.dot(evectors, nparams.reshape(npca, 1))
        verts = verts.reshape(3, n_vert) + mean_points.T
        verts *= 0.01
        opath = join(*[out_dir, f'evector_{i}.obj'])
        print(f'export file {opath}')
        export_mesh(opath, verts=verts.T, faces=faces)

        # nparams = np.zeros(npca, dtype=np.float)
        # nparams[i] = -(3*np.sqrt(evalues[0, i]))
        # verts = np.dot(evectors, nparams.reshape(npca, 1))
        # verts = verts.reshape(3, n_vert) + mean_points.T
        # verts *= 0.01
        # opath = join(*[out_dir, f'evector_{i}_1.obj'])
        # print(f'export file {opath}')
        # export_mesh(opath, verts=verts.T, faces=faces)



