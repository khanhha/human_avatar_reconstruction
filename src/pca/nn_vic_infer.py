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

def train(train_loader, valid_loader, model, criterion, optimizer, validation, args, model_dir):
    # switch to train mode
    # switch to train mode
    latest_model_path = find_latest_model_path(model_dir)

    best_model_path = os.path.join(*[model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 0
        min_val_los = 9999

    valid_losses = []
    model_type = args.model_type
    for epoch in range(epoch, args.n_epoch + 1):

        lr = adjust_learning_rate(optimizer, epoch, args.lr)

        criterion.decay_pca_weight(epoch)

        tq = tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}, lr = {lr}, pca_weight = {criterion.pca_weight}')

        losses = AverageMeter()

        model.train()
        for i, (input_f, input_s, target, height) in enumerate(train_loader):
            target_var = Variable(target).cuda()
            height_var = Variable(height).cuda()
            if model_type == 'f':
                input_f_var = Variable(input_f).cuda()
                pred = model(input_f_var, height_var)
            elif model_type == 's':
                input_s_var = Variable(input_s).cuda()
                pred = model(input_s_var, height_var)
            else:
                input_f_var = Variable(input_f).cuda()
                input_s_var = Variable(input_s).cuda()
                pred = model(input_f_var, input_s_var, height_var)

            #pred = pred.view(-1)
            #target_var  = target_var.view(-1)

            #assert (masks_probs_flat >= 0. & masks_probs_flat <= 1.).all()
            loss = criterion(pred, target_var)
            losses.update(loss, input_f.size(0))

            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_metrics = validation(model, valid_loader, criterion, args)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\n\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[model_dir, f'model_epoch_{epoch}.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)

def validate(model, val_loader, criterion, args):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input_f, input_s, target, height) in enumerate(val_loader):
            target_var = Variable(target).cuda()
            height_var = Variable(height).cuda()
            if args.model_type == 'f':
                input_f_var = Variable(input_f).cuda()
                pred = model(input_f_var, height_var)
            elif args.model_type == 's':
                input_s_var = Variable(input_s).cuda()
                pred = model(input_s_var, height_var)
            else:
                input_f_var = Variable(input_f).cuda()
                input_s_var = Variable(input_s).cuda()
                pred = model(input_f_var, input_s_var, height_var)

            loss = criterion(pred, target_var)

            losses.update(loss.item(), input_f_var.size(0))

    return {'valid_loss': losses.avg}

def load_transform(args):
    target_trans_path = os.path.join(*[args.model_root_dir, 'target_transform.jlb'])
    if not Path(target_trans_path).exists():
        target_data = load_target(args.target_dir)
        target_trans = MinMaxScaler()
        target_trans.fit(target_data)
        joblib.dump(value=target_trans, filename=target_trans_path)
    target_trans = joblib.load(filename=target_trans_path)

    height_trans_path = os.path.join(*[args.model_root_dir, 'height_transform.jlb'])
    if not Path(height_trans_path).exists():
        height_data = joblib.load(args.height_path)
        height_data = np.array([item[1] for item in height_data.items()])
        height_data = height_data.reshape(-1, 1)
        height_trans = RobustScaler()
        height_trans.fit(height_data)
        joblib.dump(value=height_trans, filename=height_trans_path)
    height_trans = joblib.load(filename=height_trans_path)

    return target_trans, height_trans

def create_test_loader(args, target_transform, height_transform):
    sil_transform = transforms.Compose([transforms.ToTensor()])

    dir_sil_f = os.path.join(*[args.sil_dir, 'sil_f'])
    dir_sil_s = os.path.join(*[args.sil_dir, 'sil_s'])

    #target_dir = None if args.target_dir == '' else args.target_dir

    heights = load_height(args.height_path)
    test_ds= ImgFullDataSet(img_transform=sil_transform,
                                  dir_f=dir_sil_f, dir_s=dir_sil_s,
                                  dir_target=args.target_dir, heights = heights,  target_transform=target_transform, height_transform=height_transform)

    test_loader= torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    return test_loader

def find_model_path(dir, hint):
    for path in Path(dir).glob('*.*'):
        if hint in str(path):
            return path
    assert False, f'no find path found with pattern {hint} found'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=False, default=None)
    ap.add_argument("-model_root_dir", type=str, required=True)
    ap.add_argument("-model_type", type=str, required=True, choices=['f', 's', 'joint'])
    ap.add_argument("-pca_model_path", type=str, required=True)
    ap.add_argument("-vic_mesh_path", type=str, required=True)
    ap.add_argument("-height_path", type=str, required=False, default='')
    ap.add_argument('-out_dir', type=str, required=True, help='output dataset directory')
    ap.add_argument('-num_classes', default=50, type=int, required=False, help='output dataset directory')

    args = ap.parse_args()

    tpl_verts, tpl_faces = import_mesh(args.vic_mesh_path)
    NV = tpl_verts.shape[0]

    model_path = find_model_path(os.path.join(*[args.model_root_dir, args.model_type]), 'best')
    if args.model_type in ['f', 's']:
        model = load_single_net(model_path, n_aux_input_feature=1)
    else:
        model = load_joint_net_161_test(model_path, num_classes=args.num_classes, n_aux_input_feature=1)

    target_transform, height_transform = load_transform(args)
    test_loader = create_test_loader(args, target_transform=target_transform, height_transform=height_transform)

    pca_model = joblib.load(filename=args.pca_model_path)
    if pca_model.whiten:
        pca_components = np.sqrt(pca_model.explained_variance_[:, np.newaxis]) * pca_model.components_
    else:
        pca_components = pca_model.components_
    pca_components = torch.Tensor(pca_components.T).cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n_tries = 1
    bar = tqdm(total=n_tries)
    cnt = 0
    for i, (input_f, input_s, target, height) in enumerate(test_loader):

        fpath= test_loader.dataset.get_filepath(i)

        target_var = Variable(target).cuda()
        height_var = Variable(height).cuda()

        if args.model_type == 'f':
            input_f_var = Variable(input_f).cuda()
            pred = model(input_f_var, height_var)
        elif args.model_type == 's':
            input_s_var = Variable(input_s).cuda()
            pred = model(input_s_var, height_var)
        else:
            input_f_var = Variable(input_f).cuda()
            input_s_var = Variable(input_s).cuda()
            pred = model(input_f_var, input_s_var, height_var)

        pred = pred.data.cpu().numpy()

        verts = pca_model.inverse_transform(pred)
        verts = verts.reshape((NV, 3))

        os.makedirs(args.out_dir, exist_ok=True)
        opath = os.path.join(*[args.out_dir, f'{fpath.stem}.obj'])
        export_mesh(opath, verts, tpl_faces)

        bar.update(1)
        cnt+=1
        if cnt > n_tries:
            break