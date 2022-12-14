import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
import scipy.io as io
from os.path import join
import os
import cv2 as cv
from sklearn.externals import joblib
import re

network_input_size = (384, 256)

class ImgFullDataSet(Dataset):
    def __init__(self, img_transform, dir_f, dir_s, dir_target, id_to_heights, target_transform = None, height_transform = None, use_input_gender = False):

        paths_f, paths_s, paths_target = self._load_names(dir_f=dir_f, dir_s=dir_s, dir_target=dir_target)
        assert len(paths_f) > 0 and len(paths_s)>0, f'empty folder {dir_f}, {dir_s}'
        self.img_transform = img_transform
        self.img_paths_f = paths_f
        self.img_paths_s = paths_s
        self.target_paths = paths_target
        self.target_transform = target_transform

        self.use_input_gender = use_input_gender
        if use_input_gender:
            self.genders = np.zeros(len(self.img_paths_f), np.uint8)
            for idx, path in enumerate(self.img_paths_f):
                if '_male' in path.stem:
                    self.genders[idx] = 1
                elif '_female' in path.stem:
                    self.genders[idx] = 0
                else:
                    assert f'no gender hint in the file name. {path.stem}'

        self.heights = []
        for path in self.img_paths_f:
            assert path.stem in id_to_heights, 'missing height'
            h = id_to_heights[path.stem]
            self.heights.append(h)

        self.heights = np.array(self.heights).astype(np.float32).reshape(-1, 1)
        self.height_transform = height_transform

        #for the case where height and target data are not available
        self.dummy_target = np.zeros(50)
        self.dummy_height = np.zeros(1)

    def _load_names(self, dir_f, dir_s, dir_target):
        #gather front and side silhouette names. for every front silhoette, the must be one corresponding side silhouette
        names = set([path.name for path in Path(dir_f).glob('*.*')])
        s_paths = []
        f_paths = []
        for name in names:
            f_path = os.path.join(*[dir_f, name])
            s_path = os.path.join(*[dir_s, name])
            if Path(f_path).exists() == True and Path(s_path).exists() == True:
                f_paths.append(Path(f_path))
                s_paths.append(Path(s_path))
            else:
                assert False, f'missing front or side silhouette : {name}'

        #TODO debug. for faster training test. comment the code after finish. the two below lines should never been included in the real traning time
        # s_paths = s_paths[:100]
        # f_paths = f_paths[:100]

        #gather the corresponding PCA target for each front-side silhouette pair
        y_paths = []
        if dir_target is not None:
            all_y_paths = dict([(path.stem, path) for path in Path(dir_target).glob('*.*')])
            for x_path in f_paths:
                assert x_path.stem in all_y_paths, f'missing target {x_path}'
                y_paths.append(all_y_paths[x_path.stem])

        return f_paths, s_paths, y_paths

    def __getitem__(self, i):
        fpath= self.img_paths_f[i]
        img_f = Image.open(fpath)
        #plt.subplot(121)
        #plt.imshow(np.asarray(img_f))
        img_f = self.img_transform(img_f)

        spath = self.img_paths_s[i]
        img_s = Image.open(spath)
        # plt.subplot(122)
        # plt.imshow(np.asarray(img_s))
        # plt.show()
        img_s = self.img_transform(img_s)

        if len(self.target_paths) > 0:
            target = np.load(self.target_paths[i]).astype(np.float32).reshape(1,-1)
            if self.target_transform is not None:
                target = self.target_transform.transform(target)
            target = target.flatten()
        else:
            #dummy value
            target = self.dummy_target

        if len(self.heights) > 0:
            aux = self.heights[i].reshape(1, -1)
            if self.height_transform is not None:
                aux = self.height_transform.transform(aux)
                aux = aux.flatten()
        else:
            aux = self.dummy_height

        if self.use_input_gender:
            aux = np.hstack([aux, self.genders[i]])

        return img_f, img_s, target, aux

    def get_filepath(self, i):
        return self.img_paths_f[i]

    def __len__(self):
        return len(self.img_paths_f)


class ImgFullDataSetPoseVariants(Dataset):
    def __init__(self, img_transform, dir_f, dir_s, dir_target, id_to_heights, target_transform=None,
                 height_transform=None, use_input_gender=False, shuffle_front_side_pairs = False):
        """
        :param img_transform:
        :param dir_f:
        :param dir_s:
        :param dir_target:
        :param id_to_heights:
        :param target_transform:
        :param height_transform:
        :param use_input_gender:
        :param n_pose_variant: number of pose variant per subject
        :param shuffle_front_side_pairs: always match front_pose_0 to side_pose_0 or the matching should be random?
        if it is random, it is assumed the ouput target of all pose_variants are the same.
        """
        print(f'create data loader (with pose variant) for folders: {dir_f}, {dir_s}, {dir_target}.')
        self.shuffle_front_side_pairs = shuffle_front_side_pairs

        paths_f, paths_s, paths_target = self._load_names(dir_f=dir_f, dir_s=dir_s, dir_target=dir_target)
        assert len(paths_f) > 0 and len(paths_s) > 0, f'empty folder {dir_f}, {dir_s}'

        self.n_pose_variant = self.get_number_of_pose_variants(paths_f)

        paths_f, paths_s, paths_target = self._sort_subject_pose_variant_names(paths_f, paths_s, paths_target, self.n_pose_variant)

        self.img_transform = img_transform
        self.img_paths_f = paths_f
        self.img_paths_s = paths_s
        self.target_paths = paths_target
        self.target_transform = target_transform

        #verification
        self.N_subject = self.count_unique_subjects(self.img_paths_f, self.img_paths_s)
        assert len(self.img_paths_f) % self.n_pose_variant == 0, 'incorrect number of files: n_file % n_pose_variant_per_file != 0'
        assert self.N_subject == int(len(self.img_paths_f)//self.n_pose_variant), 'incorrect number of files: something wrong'

        if self.shuffle_front_side_pairs == True:
            self.verify_targets(self.target_paths, self.N_subject, self.n_pose_variant)

        print(f'\tN_subject = {self.N_subject}. N files with pose variant = {len(self.img_paths_f)}')

        self.use_input_gender = use_input_gender
        if use_input_gender:
            self.genders = np.zeros(len(self.img_paths_f), np.uint8)
            for idx, path in enumerate(self.img_paths_f):
                if '_male' in path.stem:
                    self.genders[idx] = 1
                elif '_female' in path.stem:
                    self.genders[idx] = 0
                else:
                    assert f'no gender hint in the file name. {path.stem}'

        self.heights = []
        for path in self.img_paths_f:
            assert path.stem in id_to_heights, f'missing height for file name: {path.stem}'
            h = id_to_heights[path.stem]
            self.heights.append(h)

        self.heights = np.array(self.heights).astype(np.float32).reshape(-1, 1)
        self.height_transform = height_transform

        # for the case where height and target data are not available
        self.dummy_target = np.zeros(50)
        self.dummy_height = np.zeros(1)

    @classmethod
    def get_number_of_pose_variants(cls, paths):
        paths = sorted(paths)
        sample_name = remove_pose_variant_in_file_name(paths[0].name)
        n_pose = 0
        #after paths are sorted, we assume that pose variants of a subject will be continuous name0_pose0.png, name0_pose1.png, name0_pose2, ..., name0_pose30.
        for path in paths:
            name = remove_pose_variant_in_file_name(path.name)
            if name == sample_name:
                n_pose += 1
            else:
                break
        return n_pose

    @staticmethod
    def count_unique_subjects(f_paths, s_paths):
        unq_names = set()
        for fpath, spath in zip(f_paths, s_paths):
            unq_fname = remove_pose_variant_in_file_name(fpath.name)
            unq_sname = remove_pose_variant_in_file_name(spath.name)
            assert unq_fname == unq_sname, 'mismatched front and side sil names after pose variatn posfix is removed'
            unq_names.add(unq_fname)

        return len(unq_names)

    @staticmethod
    def _load_names(dir_f, dir_s, dir_target):
        # gather front and side silhouette names. for every front silhoette, the must be one corresponding side silhouette
        names = set([path.name for path in Path(dir_f).glob('*.*')])
        s_paths = []
        f_paths = []
        for name in names:
            f_path = os.path.join(*[dir_f, name])
            s_path = os.path.join(*[dir_s, name])
            if Path(f_path).exists() == True and Path(s_path).exists() == True:
                f_paths.append(Path(f_path))
                s_paths.append(Path(s_path))
            else:
                assert False, f'missing front or side silhouette : {name}'

        # TODO debug. for testing training. remember to comment the code after you're done with debugging.
        #  the two below lines should never been included in the real traning time
        # s_paths = s_paths[:100]
        # f_paths = f_paths[:100]

        # gather the corresponding PCA target for each front-side silhouette pair
        y_paths = []
        if dir_target is not None:
            all_y_paths = dict([(path.stem, path) for path in Path(dir_target).glob('*.*')])
            for x_path in f_paths:
                assert x_path.stem in all_y_paths, f'missing target {x_path}'
                y_paths.append(all_y_paths[x_path.stem])

        return f_paths, s_paths, y_paths

    @staticmethod
    def verify_targets(target_paths, N_subject, N_pose_variant):
        #check that all targets of pose variants of a subject are the same.
        for i in range(N_subject):
            j_0 = i*N_pose_variant
            j_1 = (i+1)*N_pose_variant
            target_0 = np.load(target_paths[j_0]).astype(np.float32)
            #verify that all other target of other pose variants are the same
            for j in range(j_0, j_1):
                target_j = np.load(target_paths[j]).astype(np.float32)
                same = np.allclose(target_0, target_j)
                assert same, f'targets of pose variants are different for subject {i}'

    @classmethod
    def _sort_subject_pose_variant_names(cls, f_paths, s_paths, y_paths, n_pose_variant):
        verify_pose_variants_per_name(f_paths, n_pose_variant)
        verify_pose_variants_per_name(s_paths, n_pose_variant)
        verify_pose_variants_per_name(y_paths, n_pose_variant)
        f_paths = sorted(f_paths)
        s_paths = sorted(s_paths)
        y_paths = sorted(y_paths)

        #sanity check
        for fpath,spath,ypath in zip(f_paths, s_paths, y_paths):
            assert fpath.stem == spath.stem, 'mismatched front/side name after sorting'
            assert fpath.stem == ypath.stem, 'mismathced front/target name after sorting'

        #sanity check
        assert len(f_paths)%n_pose_variant == 0
        N_subject = len(f_paths)//n_pose_variant
        for i in range(N_subject):
            #pose variants index range
            j0 = i*n_pose_variant
            j1 = (i+1)*n_pose_variant
            unq_name = remove_pose_variant_in_file_name(f_paths[j0].stem)
            #all the pose variant names of this subject must share the same subject name
            for j in range(j0, j1):
                unq_name_1 = remove_pose_variant_in_file_name(f_paths[j].stem)
                assert unq_name_1 == unq_name, 'something wrong. pose variant names of a single subject are not next to each other'

        return f_paths, s_paths, y_paths

    def __getitem__(self, subject_idx):
        """
        :param subject_idx: the index of the human subject. [0, len(img_path_f)//self.n_pose_variant]
        :return:
        """
        #select a random pose variant for this human sujbect
        i_base = subject_idx * self.n_pose_variant
        i_front = i_base + np.random.randint(0, self.n_pose_variant)
        # choose another random side pose if we're asked to. otherwise, use the same front pose index to select side pose
        i_side  = i_base + np.random.randint(0, self.n_pose_variant) if self.shuffle_front_side_pairs else i_front

        fpath = self.img_paths_f[i_front]
        img_f = Image.open(fpath)
        # plt.subplot(121)
        # plt.imshow(np.asarray(img_f))
        img_f = self.img_transform(img_f)

        spath = self.img_paths_s[i_side]
        img_s = Image.open(spath)
        # plt.subplot(122)
        # plt.imshow(np.asarray(img_s))
        # plt.show()
        img_s = self.img_transform(img_s)

        #Warning: here we assume that targets for all the front and side pose variants are the same.
        if len(self.target_paths) > 0:
            target = np.load(self.target_paths[i_front]).astype(np.float32).reshape(1, -1)
            if self.target_transform is not None:
                target = self.target_transform.transform(target)
            target = target.flatten()
        else:
            # dummy value
            target = self.dummy_target

        if len(self.heights) > 0:
            aux = self.heights[i_front].reshape(1, -1)
            if self.height_transform is not None:
                aux = self.height_transform.transform(aux)
                aux = aux.flatten()
        else:
            aux = self.dummy_height

        if self.use_input_gender:
            aux = np.hstack([aux, self.genders[i_front]])

        return img_f, img_s, target, aux

    def get_filepath(self, subject_idx):
        #TODO. how to know the actual file path that we already returned?
        idx = subject_idx * self.n_pose_variant
        path = self.img_paths_f[idx]
        unq_name = remove_pose_variant_in_file_name(path.name)
        return Path(os.path.join(*[path.parent, unq_name]))

    def __len__(self):
        #this is very important here.
        #we dont return the length of the actual file name list :img_paths_f, which include subject pose variant names.
        #here we just return the number of actual subjects, after their pose variant names are removed.
        #by doing it, it will be able to return a random pose variant each time pytorch training request for a subject image pairs.
        return self.N_subject

class ImgPairDataSet(Dataset):

    def __init__(self, img_transform, img_paths_f, img_paths_s, target_paths, target_transform):
        self.img_transform = img_transform
        self.img_paths_f = img_paths_f
        self.img_paths_s = img_paths_s
        self.target_paths = target_paths
        self.target_transform = target_transform

        assert len(self.img_paths_f) == len(self.img_paths_s)
        assert len(self.img_paths_f) == len(self.target_paths)

    def __getitem__(self, i):
        fpath= self.img_paths_f[i]
        img_f = Image.open(fpath)
        #plt.subplot(121)
        #plt.imshow(np.asarray(img_f))
        img_f = self.img_transform(img_f)

        spath = self.img_paths_s[i]
        img_s = Image.open(spath)
        # plt.subplot(122)
        # plt.imshow(np.asarray(img_s))
        # plt.show()
        img_s = self.img_transform(img_s)

        target = np.load(self.target_paths[i]).astype(np.float32)
        target = target.reshape(1, -1)
        if self.target_transform is not None:
            target = self.target_transform.transform(target)
        target = target.flatten()
        #print(target.min(), target.max())
        return img_f, img_s, target #torch.from_numpy(np.array(mask, dtype=np.int64))

    def get_filepath(self, i):
        return self.img_paths_f[i]

    def __len__(self):
        return len(self.img_paths_f)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def load_target(target_dir):
    data = []
    for path in Path(target_dir).glob('*.npy'):
        data.append(np.load(path))
    return data

def load_height(path):
    results = {}
    with open(path, 'r') as file:
        for l in file.readlines():
            name, h = l.split(' ')
            h = h.replace('\n', '')
            h = float(h)
            results[name] = h
    return results

def create_pair_loader(input_dir_f, input_dir_s, target_dir, transforms, target_transform = None, batch_size = 16, shuffle=False):
    names = set([path.name for path in Path(input_dir_f).glob('*.*')])
    s_paths = []
    f_paths = []
    for name in names:
        f_path = os.path.join(*[input_dir_f, name])
        s_path = os.path.join(*[input_dir_s, name])
        if Path(f_path).exists() == True and Path(s_path).exists() == True:
            f_paths.append(Path(f_path))
            s_paths.append(Path(s_path))
        else:
            print(f'missing front or side silhouette : {name}')

    all_y_paths = dict([(path.stem, path) for path in Path(target_dir).glob('*.*')])
    y_paths = []
    for x_path in f_paths:
        assert x_path.stem in all_y_paths, f'missing target {x_path}'
        y_paths.append(all_y_paths[x_path.stem])

    dataset =  ImgPairDataSet(transforms, img_paths_f=f_paths, img_paths_s=s_paths, target_paths= y_paths, target_transform=target_transform)

    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=shuffle, num_workers=4)

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

def reconstruct_mesh_from_pca(pca_model, x):
    n_vert = pca_model['n_vert']
    npca = pca_model['n_pca']
    evectors = pca_model['evectors']
    mean_points = pca_model['mean_points']
    faces = pca_model['faces']

    verts = np.dot(evectors, x.reshape(npca, 1))
    verts = verts.reshape(3, n_vert) + mean_points.T
    verts *= 0.01

    return verts.T, faces

def load_pca_model(model_dir, npca=100):
    faces = load_faces(f'{model_dir}/model.dat')
    mean_points  = io.loadmat(f'{model_dir}/meanShape.mat')['points'] #shape=(6449,3)
    evectors_org = io.loadmat(join(*[model_dir, 'evectors.mat']))['evectors']
    evalues_org  = io.loadmat(join(*[model_dir, 'evalues.mat']))['evalues']
    evectors = evectors_org[:npca, :].T
    evalues  = evalues_org[:,:npca]

    n_vert = 6449
    n_face = 12894
    return {'evectors':evectors, 'evalues':evalues, 'faces':faces, 'mean_points':mean_points, 'n_pca':npca, 'n_vert':n_vert, 'n_face':n_face}

def crop_silhouette(img, crop_hor = True, hor_dx = 50):
    #assert len(img.shape) == 2

    row_mask = np.sum(img, axis=1)
    row_mask = np.argwhere(row_mask)

    head_tip_y = np.min(row_mask) - 2
    toe_tip_y = np.max(row_mask) + 2
    img = img[head_tip_y:toe_tip_y, :]

    if crop_hor:
        col_mask = np.sum(img, axis=0)
        col_mask = np.argwhere(col_mask)
        left_x  = np.min(col_mask) - hor_dx
        right_x = np.max(col_mask) + hor_dx
        img = img[:, left_x:right_x]

    return img

def crop_silhouette_height(sil, mask):
    #assert len(sil.shape) == 2

    row_mask = np.sum(mask, axis=1)
    row_mask = np.argwhere(row_mask)

    head_tip_y = max(np.min(row_mask) - 1, 0)
    toe_tip_y = np.max(row_mask) + 1
    if len(sil.shape) == 2:
        sil = sil[head_tip_y:toe_tip_y, :]
    else:
        sil = sil[head_tip_y:toe_tip_y, :, :]

    return sil, (head_tip_y, toe_tip_y)

def crop_silhouette_width(sil, mask):
    col_mask = np.sum(mask, axis=0)
    col_mask = np.argwhere(col_mask)
    left_x =   max(np.min(col_mask) - 1, 0)
    right_x =  np.max(col_mask) + 1

    if len(sil.shape) == 2:
        sil = sil[:, left_x:right_x]
    else:
        sil = sil[:, left_x:right_x, :]

    return sil, (left_x, right_x)

def crop_silhouette_pair_blender(sil_f, sil_s, size):
    #plt.imshow(sil_f)
    #plt.show()
    th3, sil_f = cv.threshold(sil_f, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    th3, sil_s = cv.threshold(sil_s, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    sil_f, sil_s, _, _ = crop_silhouette_pair(sil_f, sil_s, mask_f=sil_f, mask_s=sil_s, target_h=size[0], target_w=size[1],
                                        px_height=int(0.9 * size[0]))

    th3, sil_f = cv.threshold(sil_f, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th3, sil_s = cv.threshold(sil_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return sil_f, sil_s

def crop_silhouette_pair(img_f, img_s, mask_f, mask_s, px_height = 364, target_h = 384, target_w = 256):
    Trans_f = None
    Trans_s = None
    if img_f is not None:
        img_f, y_crop_range = crop_silhouette_height(img_f, mask_f)
        img_f, x_crop_range = crop_silhouette_width(img_f, mask_f)

        T_crop = np.array([[1.0, 0.0, -x_crop_range[0]],
                           [0.0, 1.0, -y_crop_range[0]],
                           [0.0, 0.0, 1.0]], dtype=np.float32)

        h_ratio = px_height / img_f.shape[0]
        img_f = cv.resize(img_f, dsize= None, fx=h_ratio, fy=h_ratio, interpolation=cv.INTER_AREA)

        S = np.array([[h_ratio, 0.0, 0.0],
                      [0.0,     h_ratio, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        ver_ext = int((target_h - img_f.shape[0]) / 2)
        hor_ext = int((target_w - img_f.shape[1]) / 2)
        img_f = cv.copyMakeBorder(img_f, top=ver_ext, bottom=ver_ext, left=hor_ext, right=hor_ext, borderType=cv.BORDER_CONSTANT)

        T_pad = np.array([[1.0, 0.0, hor_ext],
                          [0.0, 1.0, ver_ext],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

        if img_f.shape[0] != target_h or img_f.shape[1] != target_w:
            img_f = cv.resize(img_f, dsize= (target_w, target_h), interpolation=cv.INTER_AREA)

        Trans_f = np.matmul(S, T_crop)
        Trans_f = np.matmul(T_pad, Trans_f)

    if img_s is not None:
        img_s, y_crop_range = crop_silhouette_height(img_s, mask_s)
        img_s, x_crop_range = crop_silhouette_width(img_s, mask_s)

        T_crop = np.array([[1.0, 0.0, -x_crop_range[0]],
                           [0.0, 1.0, -y_crop_range[0]],
                           [0.0, 0.0, 1.0]], dtype=np.float32)

        h_ratio = px_height / img_s.shape[0]
        img_s = cv.resize(img_s, dsize= None, fx=h_ratio, fy=h_ratio, interpolation=cv.INTER_AREA)

        S = np.array([[h_ratio, 0.0, 0.0],
                      [0.0,     h_ratio, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        ver_ext = int((target_h - img_s.shape[0]) / 2)
        hor_ext = int((target_w - img_s.shape[1]) / 2)
        img_s = cv.copyMakeBorder(img_s, top=ver_ext, bottom=ver_ext, left=hor_ext, right=hor_ext, borderType=cv.BORDER_CONSTANT)

        T_pad = np.array([[1.0, 0.0, hor_ext],
                          [0.0, 1.0, ver_ext],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

        #assert sil_s.shape[0] == sil_f.shape[0]

        #for sure
        if img_s.shape[0] != target_h or img_s.shape[1] != target_w:
            img_s = cv.resize(img_s, dsize= (target_w, target_h), interpolation=cv.INTER_AREA)

        Trans_s = np.matmul(S, T_crop)
        Trans_s = np.matmul(T_pad, Trans_s)

    # plt.axes().set_aspect(1.0)
    # plt.subplot(121)
    # plt.imshow(sil_f)
    # plt.subplot(122)
    # plt.imshow(sil_s)
    # plt.show()

    return img_f, img_s, Trans_f, Trans_s

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
        model_paths.append(path)
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)

    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = np.argmax(epochs)
        return model_paths[max_idx]
    else:
        return None


def remove_pose_variant_in_file_name(name):
    parts = name.split('_')
    pose_part = None
    for part in parts:
        if 'pose' in part:
            pose_part = part
            break
    assert pose_part is not None, 'incorrect file name format. missing pose hint name'
    new_name = name.replace(pose_part, '')
    return new_name

from collections import defaultdict
def verify_pose_variants_per_name(paths, N_pose):
    """
    verify that there are extact N_pose variants per human subject
    :param paths:
    :param N_pose:
    """

    #extract all the unique subject names [human0_pose1.*, human0_pose30] => human0
    unique_names = set()
    for path in paths:
        unique_names.add(remove_pose_variant_in_file_name(Path(path).stem))

    #count the number of pose variants per subject
    counter = defaultdict(int)
    for path in paths:
        unq_name = remove_pose_variant_in_file_name(Path(path).stem)
        counter[unq_name] += 1

    #assert that there are exact the expected number of pose per subject
    invalid = True
    for k, value in counter.items():
        if value != N_pose:
            invalid = False
            print(f'missing pose variants for object file name {k}. n_pose = {value} while it should be {N_pose}')
    assert invalid, 'pose_verification_result: failed'
