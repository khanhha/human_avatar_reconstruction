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
from pca.dense_net import load_joint_net_161_test
from pca.nn_util import load_pca_model, reconstruct_mesh_from_pca, create_pair_loader, create_pair_loader_inference, network_input_size, crop_silhouette

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir", type=str, required=True)
    ap.add_argument("-sil_s_dir", type=str, required=True)
    ap.add_argument("-vic_mesh_path", type=str, required=True)
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-target_transform_path", type=str, required=False, default='')
    ap.add_argument("-target_dir", type=str, required=False, default='')
    ap.add_argument("-pca_model_path", type=str, required=True)
    ap.add_argument("-out_dir", type=str, required=True)
    ap.add_argument("-n_tries", type=int, required=False, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in Path(args.out_dir).glob('*.obj'):
        os.remove(path)

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)
    NV = tpl_verts.shape[0]

    model = load_joint_net_161_test(args.model_path, num_classes=50)
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

    target_dir = None if args.target_dir == '' else args.target_dir
    loader = create_pair_loader_inference(args.sil_f_dir, args.sil_s_dir, transform, batch_size=1, target_dir=target_dir)
    #loader = create_pair_loader(args.sil_f_dir, args.sil_s_dir, args.target_dir, transform, target_transform)

    if args.n_tries == -1:
        n_tries = len(loader.dataset)
    else:
        n_tries = args.n_tries

    bar = tqdm(total=n_tries)
    cnt = 0
    for i, (input_f, input_s, _) in enumerate(loader):
        input_f_var = Variable(input_f).cuda()
        input_s_var = Variable(input_s).cuda()

        pred = model(input_f_var, input_s_var)
        pred = pred.data.cpu().numpy()

        if target_transform is not None:
            pred = target_transform.inverse_transform(pred)

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


# mobile images profile
# -sil_f_dir
# /home/khanhhh/data_1/projects/Oh/data/mobile_image_silhouettes/sil_f
# -sil_s_dir
# /home/khanhhh/data_1/projects/Oh/data/mobile_image_silhouettes/sil_s
# -model_path
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/models/model_vic_nowhiten/model_best.pt
# -pca_model_path
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_model.jlb
# -vic_mesh_path
# /media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj
# -out_dir
# /home/khanhhh/data_1/projects/Oh/data/mobile_image_silhouettes/mesh_result_vic
# -target_transform_path
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/models/pca_vic_target_transform.jlb

# caesar images profile
# -sil_f_dir
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_f/test
# -sil_s_dir
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_s/test
# -model_path
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/models/model_vic_nowhiten/model_best.pt
# -pca_model_path
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_model.jlb
# -vic_mesh_path
# /media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj
# -out_dir
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/result/model_vic_1
# -target_transform_path
# /home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/models/pca_vic_target_transform.jlb

