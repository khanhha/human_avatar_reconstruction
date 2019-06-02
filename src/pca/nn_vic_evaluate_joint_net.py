import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pca.dense_net import densenet121
import argparse
import os
from pathlib import Path
import pickle
from PIL import Image
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA
from os.path import join
import scipy.io as io
from common.obj_util import export_mesh, import_mesh_obj
from pca.dense_net import load_joint_net_161_test
from pca.nn_util import load_pca_model, reconstruct_mesh_from_pca, create_pair_loader, create_pair_loader_inference, network_input_size, crop_silhouette

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir", type=str, required=True)
    ap.add_argument("-sil_s_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=False, default='')
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-target_transform_path", type=str, required=False, default='')
    args = ap.parse_args()

    model = load_joint_net_161_test(args.model_path, num_classes=50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_size = network_input_size
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    target_transform = None
    if args.target_transform_path != '':
        target_transform = joblib.load(filename=args.target_transform_path)
        print(f'used target_transform: {target_transform}' )

    target_dir = None if args.target_dir == '' else args.target_dir
    bs = 8
    loader = create_pair_loader(args.sil_f_dir, args.sil_s_dir, args.target_dir, transform, target_transform = target_transform, batch_size=bs)

    n_files = len(loader.dataset)
    bar = tqdm(total=len(loader))
    cnt = 0
    error = 0.0
    for i, (input_f, input_s, target) in enumerate(loader):
        input_f_var = Variable(input_f).cuda()
        input_s_var = Variable(input_s).cuda()

        pred = model(input_f_var, input_s_var)
        pred = pred.data.cpu().numpy()

        target = target.data.cpu().numpy()
        e = np.sum(np.square(pred - target))
        error += e

        bar.update(bs)

    bar.close()

    error /= float(n_files)

    print(f'total error = {error}')
