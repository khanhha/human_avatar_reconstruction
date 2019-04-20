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
from common.obj_util import export_mesh, import_mesh
from pca.dense_net import load_single_net
from pca.nn_util import load_pca_model, reconstruct_mesh_from_pca, create_single_loader, create_pair_loader_inference, network_input_size, crop_silhouette

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_dir", type=str, required=True)
    ap.add_argument("-vic_mesh_path", type=str, required=True)
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-pca_model_path", type=str, required=True)
    ap.add_argument("-target_transform_path", type=str, required=False, default='')
    ap.add_argument("-target_dir", type=str, required=False, default='')
    ap.add_argument("-out_dir", type=str, required=True)
    ap.add_argument("-n_tries", type=int, required=False, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in Path(args.out_dir).glob('*.obj'):
        os.remove(str(path))

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)
    NV = tpl_verts.shape[0]

    model = load_single_net(args.model_path, num_classes=50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_size = network_input_size
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pca_model = joblib.load(filename=args.pca_model_path)

    target_transform = None
    if args.target_transform_path != '':
        target_transform = joblib.load(filename=args.target_transform_path)
        print(f'used target_transform: {target_transform}' )

    os.makedirs(args.out_dir, exist_ok=True)

    loader = create_single_loader(args.sil_dir, args.target_dir, transform, target_transform, batch_size=1)

    if args.n_tries == -1:
        n_tries = len(loader.dataset)
    else:
        n_tries = args.n_tries

    bar = tqdm(total=n_tries)
    cnt = 0
    for i, (input, target) in enumerate(loader):
        input_var = Variable(input).cuda()

        pred = model(input_var)
        pred = pred.data.cpu().numpy()

        if target_transform is not None:
            pred = target_transform.inverse_transform(pred)

        target = target.data.cpu().numpy()
        diff = np.abs(pred - target)

        verts = pca_model.inverse_transform(pred)
        verts = verts.reshape((NV, 3))

        fpath= loader.dataset.get_filepath(i)
        opath = join(*[args.out_dir, f'{fpath.stem}.obj'])
        export_mesh(opath, verts, tpl_faces)

        bar.update(1)
        cnt+=1
        if cnt > n_tries:
            break
    bar.close()
