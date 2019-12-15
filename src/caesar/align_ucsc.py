import numpy as np
from common.obj_util import import_mesh_obj, export_mesh
from common.transformations import  affine_matrix_from_points, scale_matrix, translation_matrix
from pathlib import Path
import pickle
from tqdm import tqdm
import os
mpii_landmark_idxs = [1280,4411] + [1365,4496] + [1301,4432] + [1701,4831] + \
            [1093] + \
            [2297,5428] + \
            [1110] + \
            [29] + \
            [648,3865] + \
            [44] + \
            [927,4136] + \
            [68] + \
            [812,4028]

ucsc_landmark_idxs = [588,635] + [983,1022] + [1030,1087] + [2707,2737] + \
            [12498] + \
            [6815,6766] + \
            [6889] + \
            [7214] + \
            [9172,9285] + \
            [9251] + \
            [10713,10777] + \
            [10557] + \
            [8438,8517]

def transform_verts(V, T):
    assert V.shape[1] == 3
    V = np.hstack([V, np.ones((V.shape[0],1))])
    V = np.dot(T, V.T).T
    return V[:,:3]

def estimate_rigid_alignment_transforms(ucsc_verts, mpii_verts):
    # transform ucsc closer to mpii mesh to
    # make it easier for rigid transformation estimation.
    #ucsc_verts *= 10.0
    #ucsc_verts[:,2] += 10
    T0 = translation_matrix((0.0,0.0,10.0)) * scale_matrix(10.0)
    ucsc_verts_0 = transform_verts(ucsc_verts, T0)

    ucsc_lds = ucsc_verts_0[ucsc_landmark_idxs, :]
    mpii_lds = mpii_verts[mpii_landmark_idxs, :]
    T1 = affine_matrix_from_points(ucsc_lds.T, mpii_lds.T, shear=False, scale=False)

    #ucsc_lds_1 = np.hstack([ucsc_lds, np.ones((ucsc_lds.shape[0],1))])
    #mpii_lds_1 = np.hstack([mpii_lds, np.ones((mpii_lds.shape[0],1))])

    #ucsc_verts_1 = transform_verts(ucsc_verts_0, T1)
    #ucsc_mean_out_path = "/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/body_alignment/ucsc_test_transformed.obj"
    #export_mesh(ucsc_mean_out_path, verts=ucsc_verts_1, faces=ucsc_faces)

    return T0, T1

def align_ucsc(dir_in, dir_out, T, faces):
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    paths = [path for path in dir_in.glob('*.pkl')]
    for path in tqdm(paths, desc="ucsc"):
        with open(str(path), 'rb') as file:
            verts = pickle.load(file)
        verts_1 = transform_verts(verts, T)

        with open(dir_out/f'ucsc_{path.stem}.pkl', 'wb') as file:
            pickle.dump(obj=verts_1, file=file)
        if np.random.rand()< 0.1:
            export_mesh(dir_out/f'ucsc_{path.stem}.obj', verts=verts_1, faces=faces)

def align_mpii(dir_in, dir_out, T, faces):
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    paths = [path for path in dir_in.glob('*.pkl')]
    for path in tqdm(paths, desc="mpii"):
        with open(str(path), 'rb') as file:
            verts = pickle.load(file)
        verts_1 = transform_verts(verts, T)
        with open(dir_out/f'mpii_{path.stem}.pkl', 'wb') as file:
            pickle.dump(obj=verts_1, file=file)
        if np.random.rand()< 0.1:
            export_mesh(dir_out/f'mpii_{path.stem}.obj', verts=verts_1, faces=faces)

def calc_mpii_mean_mesh(dir):
    dir = Path(dir)
    paths = [path for path in dir.glob('*.obj')]
    mean = None
    for path in tqdm(paths):
        verts, _ = import_mesh_obj(path)
        if mean is None:
            mean = np.copy(verts)
        else:
            mean += verts
    mean = mean/len(paths)
    return mean

def calc_ucsc_mean_mesh(dir_male, dir_female):
    dir_male = Path(dir_male)
    paths_male = [path for path in dir_male.glob('*.obj')]
    mean = None
    for path in tqdm(paths_male):
        verts, _ = import_mesh_obj(path)
        if mean is None:
            mean = np.copy(verts)
        else:
            mean += verts

    dir_female = Path(dir_female)
    paths_female = [path for path in dir_female.glob('*.obj')]
    for path in tqdm(paths_female):
        verts, _ = import_mesh_obj(path)
        if mean is None:
            mean = np.copy(verts)
        else:
            mean += verts

    mean = mean / (len(paths_male) + len(paths_female))

    return mean

if __name__ == '__main__':
    ucsc_mean_path = "/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/mean_male_ucsc.obj"
    mpii_mean_path = "/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/mpii_mean_mesh_centered.obj"
    ucsc_verts, ucsc_faces = import_mesh_obj(ucsc_mean_path)
    mpii_verts, mpii_faces = import_mesh_obj(mpii_mean_path)

    #this is created manually using Blender. manually scale, rotate and translate the mean UCSC to match the standard mean MPII
    T = (((6.2254133224487305, -7.874216079711914, -2.73009117535139e-08, -0.09876736253499985),
          (7.87327766418457, 6.224671363830566, 0.15500684082508087, -1.0474116802215576),
          (-0.1215951070189476, -0.09613402187824249, 10.036684036254883, 10.017744064331055),
          (0.0, 0.0, 0.0, 1.0)))
    T = np.array(T)

    vic_mesh_path = '/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/vic_origin_mesh.obj'
    _, vic_faces = import_mesh_obj(vic_mesh_path)

    root_out_dir = Path("/media/F/projects/Oh/data/victoria_ceasar/")
    out_dir =  root_out_dir/"aligned_mesh"
    os.makedirs(out_dir, exist_ok=True)

    ucsc_female_dir = "/media/D1/data_1/projects/Oh/data/3d_human/victoria_caesar/ucsc_female/"
    align_ucsc(dir_in=ucsc_female_dir, dir_out=out_dir, T=T,faces=vic_faces)

    ucsc_male_dir = "/media/D1/data_1/projects/Oh/data/3d_human/victoria_caesar/ucsc_male/"
    align_ucsc(dir_in=ucsc_male_dir, dir_out=out_dir, T=T, faces=vic_faces)

    female_paths = [f'ucsc_{path.stem}' for path in Path(ucsc_female_dir).glob("*.pkl")]
    female_names = set(female_paths)

    #mpii
    #this is created manually using Blender. manually scale, rotate and translate the real mean MPII to the centered MPII
    T_mpii = (((0.010240868665277958, 0.0, 0.0, 0.0),
          (0.0, 0.010240868665277958, 0.0, -1.5434037446975708),
          (0.0, 0.0, 0.010240868665277958, -0.4442373812198639),
          (0.0, 0.0, 0.0, 1.0)))
    T_mpii = np.array(T_mpii)
    mpii_dir = "/media/D1/data_1/projects/Oh/data/3d_human/victoria_caesar/mpii/"
    align_mpii(dir_in=mpii_dir, dir_out=out_dir, T=T_mpii,faces=vic_faces)

    # output female file namess
    mpii_female_name_path = "/media/D1/data_1/projects/Oh/data/3d_human/caesar_obj/female_names.txt"
    with open(mpii_female_name_path, 'r') as file:
        for line in file.readlines():
            name = line.replace('\n', '')
            assert '.obj' not in name, 'incorrect name format'
            female_names.add(f'mpii_{name}')

    with open(root_out_dir/"female_names.txt", 'wt') as file:
        for name in female_names:
            file.write(f'{name}\n')

