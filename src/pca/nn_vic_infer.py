import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pca.dense_net import densenet121, load_joint_net_161_test, load_single_net
import argparse
import os
from pathlib import Path
import pickle
from PIL import Image
from tqdm import tqdm
from common.obj_util import export_mesh, import_mesh
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.externals import joblib
from pca.dense_net import JointMask
from pca.nn_util import  AverageMeter, load_target, ImgFullDataSet, load_height
from pca.nn_util import create_pair_loader, find_latest_model_path, load_pca_model, adjust_learning_rate, network_input_size
from pca.losses import SMPLLoss
from pca.nn_vic_model import  NNModelWrapper


def create_test_loader(args, target_transform, height_transform):
    sil_transform = transforms.Compose([transforms.ToTensor()])

    dir_sil_f = args.sil_f_dir #os.path.join(*[args.sil_dir, 'sil_f'])
    dir_sil_s = args.sil_s_dir #os.path.join(*[args.sil_dir, 'sil_s'])

    #target_dir = None if args.target_dir == '' else args.target_dir

    heights = load_height(args.height_path)
    test_ds= ImgFullDataSet(img_transform=sil_transform,
                                  dir_f=dir_sil_f, dir_s=dir_sil_s,
                                  dir_target=args.target_dir, id_to_heights=heights,  target_transform=target_transform, height_transform=height_transform)

    test_loader= torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    return test_loader

def find_model_path(dir, hint):
    for path in Path(dir).glob('*.*'):
        if hint in str(path):
            return path
    assert False, f'no find path found with pattern {hint} found'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir", type=str, required=True)
    ap.add_argument("-sil_s_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=False, default=None)
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-vic_mesh_path", type=str, required=True)
    ap.add_argument("-height_path", type=str, required=False, default='')
    ap.add_argument('-out_dir', type=str, required=True, help='output dataset directory')
    ap.add_argument('-num_classes', default=50, type=int, required=False, help='output dataset directory')

    args = ap.parse_args()

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)
    NV = tpl_verts.shape[0]

    model_wrapper = NNModelWrapper.load(args.model_path)

    test_loader = create_test_loader(args, target_transform=model_wrapper.pca_target_transform, height_transform=model_wrapper.height_transform)

    model = model_wrapper.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n_tries = 1
    bar = tqdm(total=n_tries)
    cnt = 0
    for i, (input_f, input_s, target, height) in enumerate(test_loader):

        fpath= test_loader.dataset.get_filepath(i)

        if 'syn_' in fpath.name:
            continue

        target_var = Variable(target).cuda()
        height_var = Variable(height).cuda()

        if model_wrapper.model_type == 'f':
            input_f_var = Variable(input_f).cuda()
            pred = model(input_f_var, height_var)
        elif model_wrapper.model_type == 's':
            input_s_var = Variable(input_s).cuda()
            pred = model(input_s_var, height_var)
        else:
            input_f_var = Variable(input_f).cuda()
            input_s_var = Variable(input_s).cuda()
            pred = model(input_f_var, input_s_var, height_var)

        pred = pred.data.cpu().numpy()

        if model_wrapper.pca_target_transform is not None:
            pred = model.pca_target_transform.inverse_transform(pred)

        verts = model_wrapper.pca_model.inverse_transform(pred)
        verts = verts.reshape((NV, 3))

        os.makedirs(args.out_dir, exist_ok=True)
        opath = os.path.join(*[args.out_dir, f'{fpath.stem}.obj'])
        export_mesh(opath, verts, tpl_faces)

        bar.update(1)
        cnt+=1
        if cnt > n_tries:
            break