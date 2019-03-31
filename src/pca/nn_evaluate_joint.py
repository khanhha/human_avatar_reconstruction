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
import pickle
from tqdm import tqdm
from os.path import join
import scipy.io as io
from common.obj_util import export_mesh, import_mesh
from pca.dense_net import load_joint_net_161_test
from pca.nn_util import load_pca_model, reconstruct_mesh_from_pca, create_pair_loader, create_pair_loader_inference
from pca.losses import SMPLLoss

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir", type=str, required=True)
    ap.add_argument("-sil_s_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=True)
    ap.add_argument("-target_scaler_path", type=str, required=True)
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-pca_model_dir", type=str, required=True)
    ap.add_argument("-n_classes", type=int, default=50, required=False)
    args = ap.parse_args()

    model = load_joint_net_161_test(args.model_path, num_classes=args.n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_size = 224
    transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(args.target_scaler_path, 'rb') as file:
        target_scaler = pickle.load(file)

    pca_evectors = load_pca_model(args.pca_model_dir, npca=args.n_classes)['evectors']
    pca_evectors = torch.Tensor(pca_evectors).cuda()

    loader = create_pair_loader(args.sil_f_dir, args.sil_s_dir, args.target_dir, transforms=transform , target_transform=target_scaler, batch_size=16)

    loss_0_criterio = SMPLLoss(pca=pca_evectors, pca_weight=1.0)
    loss_1_criterio = SMPLLoss(pca=pca_evectors, pca_weight=0.0)
    cnt = 0
    losses_0 = []
    losses_1 = []
    for i, (input_f, input_s, target) in enumerate(loader):
        input_f_var = Variable(input_f).cuda()
        input_s_var = Variable(input_s).cuda()
        target_var = Variable(target).cuda()

        pred = model(input_f_var, input_s_var)

        l0 = loss_0_criterio(pred, target_var)
        l1 = loss_1_criterio(pred, target_var)

        losses_0.append(float(l0))

        losses_1.append(float(l1))

    print(f'loss_0 = {np.mean(losses_0)}')
    print(f'loss_1 = {np.mean(losses_1)}')
