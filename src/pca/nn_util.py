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
    def __init__(self, img_transform, img_paths, target_paths, target_transform):
        self.img_transform = img_transform
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.target_transform = target_transform
        assert len(self.img_paths) == len(self.target_paths)

    def __getitem__(self, i):
        fpath= self.img_paths[i]
        img = Image.open(fpath)
        img = self.img_transform(img)

        target = np.load(self.target_paths[i]).astype(np.float32)
        target = self.target_transform.transform(target.reshape(1,-1))
        target = target.flatten()
        #print(target.min(), target.max())
        return img, target #torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_paths)


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
        target = self.target_transform.transform(target.reshape(1,-1))
        target = target.flatten()
        #print(target.min(), target.max())
        return img_f, img_s, target #torch.from_numpy(np.array(mask, dtype=np.int64))

    def get_filepath(self, i):
        return self.img_paths_f[i]

    def __len__(self):
        return len(self.img_paths_f)

class ImgPairDataSet_FName(Dataset):
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
        img_f = self.img_transform(img_f)

        spath = self.img_paths_s[i]
        img_s = Image.open(spath)
        img_s = self.img_transform(img_s)

        target = np.load(self.target_paths[i]).astype(np.float32)
        target = self.target_transform.transform(target.reshape(1,-1))
        target = target.flatten()
        #print(target.min(), target.max())
        return img_f, img_s, target, i #torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_paths_f)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_target(target_dir):
    data = []
    for path in Path(target_dir).glob('*.npy'):
        data.append(np.load(path))
    return data

def create_pair_loader(input_dir_f, input_dir_s, target_dir, transforms, target_transform, batch_size = 16, shuffle=False):
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
        assert x_path.stem in all_y_paths
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
    return {'evectors':evectors, 'evalues':evalues, 'faces':faces, 'mean_points':mean_points, 'n_pca':100, 'n_vert':n_vert, 'n_face':n_face}
