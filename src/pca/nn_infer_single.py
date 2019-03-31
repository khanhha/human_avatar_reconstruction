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
from os.path import join
import scipy.io as io
from common.obj_util import export_mesh, import_mesh
from pca.nn_util import load_pca_model, reconstruct_mesh_from_pca

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-input_dir", type=str, required=True)
    ap.add_argument("-target_scaler_path", type=str, required=True)
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-pca_model_dir", type=str, required=True)
    ap.add_argument("-out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('hello')
    input_size = 224
    num_classes = 100
    model = densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model.to(device)

    model.eval()
    if Path(args.model_path).exists():
        state = torch.load(args.model_path)
        model.load_state_dict(state['model'])

    transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            #transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(args.target_scaler_path, 'rb') as file:
        target_scaler = pickle.load(file)

    pca_model = load_pca_model(args.pca_model_dir)

    cnt = 0

    for path in tqdm(Path(args.input_dir).glob('*.*')):
        img = Image.open(path)
        img = transform(img)
        img = Variable(img.unsqueeze(0)).cuda()  # [N, 1, H, W]
        pred = model(img)
        pred = pred.data.cpu().numpy()
        pred = target_scaler.inverse_transform(pred)
        verts, faces = reconstruct_mesh_from_pca(pca_model, pred)

        opath = join(*[args.out_dir, f'{path.stem}.obj'])
        export_mesh(opath, verts, faces)

        cnt+=1
        if cnt > 10:
            break
