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
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.externals import joblib
import re

network_input_size = (384, 256)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ImgDataSet(Dataset):
    def __init__(self, img_transform, img_paths, target_paths, target_transform=None):
        self.img_transform = img_transform
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.target_transform = target_transform
        assert len(self.img_paths) == len(self.target_paths)

    def __getitem__(self, i):
        fpath= self.img_paths[i]
        img = Image.open(fpath)
        img = self.img_transform(img)

        # plt.imshow(np.asarray(img))
        # plt.show()

        target = np.load(self.target_paths[i]).astype(np.float32)
        if self.target_transform is not None:
            target = self.target_transform.transform(target.reshape(1,-1))
        target = target.flatten()
        #print(target.min(), target.max())
        return img, target #torch.from_numpy(np.array(mask, dtype=np.int64))

    def get_filepath(self, i):
        return self.img_paths[i]

    def __len__(self):
        return len(self.img_paths)

class ImgFullDataSet(Dataset):
    def __init__(self, img_transform, dir_f, dir_s, dir_target, id_to_heights, target_transform = None, height_transform = None):

        paths_f, paths_s, paths_target = self._load_names(dir_f=dir_f, dir_s=dir_s, dir_target=dir_target)
        assert len(paths_f) > 0 and len(paths_s)>0, f'empty folder {dir_f}, {dir_s}'
        self.img_transform = img_transform
        self.img_paths_f = paths_f
        self.img_paths_s = paths_s
        self.target_paths = paths_target
        self.target_transform = target_transform

        self.heights = []
        for path in self.img_paths_f:
            assert path.stem in id_to_heights, 'missing height'
            h = id_to_heights[path.stem]
            self.heights.append(h)

        self.heights = np.array(self.heights).astype(np.float32).reshape(-1, 1)
        self.height_transform = height_transform

        self.dummy_target = np.zeros(50)
        self.dummy_height = np.zeros(1)

    def _load_names(self, dir_f, dir_s, dir_target):
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

        #TODO debug. for faster epoch. comment the code after finish
        # s_paths = s_paths[:3000]
        # f_paths = f_paths[:3000]

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
            h = self.heights[i].reshape(1, -1)
            if self.height_transform is not None:
                h = self.height_transform.transform(h)
                h = h.flatten()
        else:
            h = self.dummy_height

        return img_f, img_s, target, h

    def get_filepath(self, i):
        return self.img_paths_f[i]

    def __len__(self):
        return len(self.img_paths_f)


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

class ImgPairDataSetInfer(Dataset):
    def __init__(self, img_transform, img_paths_f, img_paths_s, target_paths = None, crop = None):
        self.img_transform = img_transform
        self.crop = crop
        self.img_paths_f = img_paths_f
        self.img_paths_s = img_paths_s
        self.target_paths = target_paths
        assert len(self.img_paths_f) == len(self.img_paths_s)


    def __getitem__(self, i):
        fpath= self.img_paths_f[i]
        img_f = Image.open(fpath)
        #if self.crop is not None:
            #img_f = self.crop(np.asarray(img_f))
            #img_f = Image.fromarray(img_f)

        img_f = self.img_transform(img_f)

        # plt.subplot(121)
        # plt.imshow(img_f)

        spath = self.img_paths_s[i]
        img_s = Image.open(spath)
        #if self.crop is not None:
            #img_s = self.crop(np.asarray(img_s))
            #img_s = Image.fromarray(img_s)

        img_s = self.img_transform(img_s)

        #plt.subplot(122)
        #plt.imshow(img_s)
        #plt.show()

        target = -1
        if self.target_paths is not None:
            target = np.load(self.target_paths[i]).astype(np.float32)

        return img_f, img_s, target

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

def create_pair_loader_inference(input_dir_f, input_dir_s, transforms, target_dir = None, batch_size = 16, shuffle=False):
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

    y_paths = None
    if target_dir is not None:
        all_y_paths = dict([(path.stem, path) for path in Path(target_dir).glob('*.*')])
        y_paths = []
        for x_path in f_paths:
            assert x_path.stem in all_y_paths, f'missing target {x_path}'
            y_paths.append(all_y_paths[x_path.stem])

    dataset =  ImgPairDataSetInfer(transforms, img_paths_f=f_paths, img_paths_s=s_paths, crop=crop_silhouette, target_paths=y_paths)

    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=shuffle, num_workers=4)

def create_single_loader(input_dir, target_dir, transforms, target_transform = None, batch_size = 16):
    x_paths = [path for path in Path(input_dir).glob('*.*')]

    all_y_paths = dict([(path.stem, path) for path in Path(target_dir).glob('*.*')])
    y_paths = []
    for x_path in x_paths:
        assert x_path.stem in all_y_paths
        y_paths.append(all_y_paths[x_path.stem])

    dataset =  ImgDataSet(transforms, x_paths, y_paths, target_transform=target_transform)

    return torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=4)

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
    assert len(img.shape) == 2

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
    assert len(sil.shape) == 2

    row_mask = np.sum(mask, axis=1)
    row_mask = np.argwhere(row_mask)

    head_tip_y = max(np.min(row_mask) - 1, 0)
    toe_tip_y = np.max(row_mask) + 1

    sil = sil[head_tip_y:toe_tip_y, :]

    return sil

def crop_silhouette_width(sil, mask):
    col_mask = np.sum(mask, axis=0)
    col_mask = np.argwhere(col_mask)
    left_x =   max(np.min(col_mask) - 1, 0)
    right_x =  np.max(col_mask) + 1

    sil = sil[:, left_x:right_x]

    return sil

def crop_silhouette_pair_blender(sil_f, sil_s, size):

    th3, sil_f = cv.threshold(sil_f, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    th3, sil_s = cv.threshold(sil_s, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    sil_f, sil_s = crop_silhouette_pair(sil_f, sil_s, mask_f=sil_f, mask_s=sil_s, target_h=size[0], target_w=size[1],
                                        px_height=int(0.9 * size[0]))

    th3, sil_f = cv.threshold(sil_f, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th3, sil_s = cv.threshold(sil_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return sil_f, sil_s

def crop_silhouette_pair(img_f, img_s, mask_f, mask_s, px_height = 364, target_h = 384, target_w = 256):
    img_f = crop_silhouette_height(img_f, mask_f)
    img_s = crop_silhouette_height(img_s, mask_s)
    img_f = crop_silhouette_width(img_f, mask_f)
    img_s = crop_silhouette_width(img_s, mask_s)

    h_ratio = px_height / img_f.shape[0]
    img_f = cv.resize(img_f, dsize= None, fx=h_ratio, fy=h_ratio, interpolation=cv.INTER_AREA)
    h_ratio = px_height / img_s.shape[0]
    img_s = cv.resize(img_s, dsize= None, fx=h_ratio, fy=h_ratio, interpolation=cv.INTER_AREA)

    ver_ext = int((target_h - img_f.shape[0]) / 2)
    hor_ext = int((target_w - img_f.shape[1]) / 2)
    img_f = cv.copyMakeBorder(img_f, top=ver_ext, bottom=ver_ext, left=hor_ext, right=hor_ext, borderType=cv.BORDER_CONSTANT)

    ver_ext = int((target_h - img_s.shape[0]) / 2)
    hor_ext = int((target_w - img_s.shape[1]) / 2)
    img_s = cv.copyMakeBorder(img_s, top=ver_ext, bottom=ver_ext, left=hor_ext, right=hor_ext, borderType=cv.BORDER_CONSTANT)

    #assert sil_s.shape[0] == sil_f.shape[0]

    #for sure
    if img_f.shape[0] != target_h or img_f.shape[1] != target_w:
        img_f = cv.resize(img_f, dsize= (target_w, target_h), interpolation=cv.INTER_AREA)

    if img_s.shape[0] != target_h or img_s.shape[1] != target_w:
        img_s = cv.resize(img_s, dsize= (target_w, target_h), interpolation=cv.INTER_AREA)

    # plt.axes().set_aspect(1.0)
    # plt.subplot(121)
    # plt.imshow(sil_f)
    # plt.subplot(122)
    # plt.imshow(sil_s)
    # plt.show()

    return img_f, img_s

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