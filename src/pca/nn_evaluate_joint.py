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
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from os.path import join
import scipy.io as io
from common.obj_util import export_mesh, import_mesh
from pca.dense_net import load_joint_net_161_test
from pca.nn_util import load_pca_model, reconstruct_mesh_from_pca, create_pair_loader

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir", type=str, required=True)
    ap.add_argument("-sil_s_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=True)
    ap.add_argument("-target_scaler_path", type=str, required=True)
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-pca_model_dir", type=str, required=True)
    ap.add_argument("-out_dir", type=str, required=True)
    ap.add_argument("-on_test_set", type=bool, default=False, required=False)
    ap.add_argument("-n_classes", type=int, default=50, required=False)
    args = ap.parse_args()


    model = load_joint_net_161_test(args.model_path, num_classes=args.n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_size = 224
    transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            #transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(args.target_scaler_path, 'rb') as file:
        target_scaler = pickle.load(file)

    pca_model = load_pca_model(args.pca_model_dir, npca=args.n_classes)

    if args.on_test_set:
        sil_f_dir = os.path.join(*[args.sil_f_dir, 'test'])
        sil_s_dir = os.path.join(*[args.sil_s_dir, 'test'])
        loader = create_pair_loader(sil_f_dir, sil_s_dir, args.target_dir, transform, target_scaler, batch_size=1)

        out_mesh_dir = os.path.join(*[args.out_dir, 'test'])
    else:
        sil_f_dir = os.path.join(*[args.sil_f_dir, 'train'])
        sil_s_dir = os.path.join(*[args.sil_s_dir, 'train'])
        loader = create_pair_loader(sil_f_dir, sil_s_dir, args.target_dir, transform, target_scaler, batch_size=1)

        out_mesh_dir = os.path.join(*[args.out_dir, 'train'])

    os.makedirs(out_mesh_dir, exist_ok=True)

    cnt = 0
    for i, (input_f, input_s, target) in enumerate(loader):
        input_f_var = Variable(input_f).cuda()
        input_s_var = Variable(input_s).cuda()

        pred = model(input_f_var, input_s_var)
        pred = pred.data.cpu().numpy()
        pred = target_scaler.inverse_transform(pred)
        verts, faces = reconstruct_mesh_from_pca(pca_model, pred)

        fpath= loader.dataset.get_filepath(i)
        opath = join(*[out_mesh_dir, f'{fpath.stem}.obj'])
        export_mesh(opath, verts, faces)

        cnt += 1
        if cnt > 200:
            break