import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pca.dense_net import densenet121, load_joint_net_161_train
import argparse
import os
from pathlib import Path
import pickle
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from pca.dense_net import JointMask
from pca.nn_util import  ImgPairDataSet, AverageMeter, load_target
from pca.nn_util import create_pair_loader

def train(train_loader, valid_loader, model, criterion, optimizer, validation, args):
    # switch to train mode
    model.train()
    if Path(args.model_path).exists():
        state = torch.load(args.model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}'.format(epoch))
    else:
        epoch = 0

    valid_losses = []
    min_val_los = 9999
    for epoch in range(epoch, args.n_epoch + 1):

        #adjust_learning_rate(optimizer, epoch)

        tq = tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        model.train()
        for i, (input_f, input_s, target) in enumerate(train_loader):
            input_f_var  = Variable(input_f).cuda()
            input_s_var  = Variable(input_s).cuda()

            target_var = Variable(target).cuda()

            pred = model(input_f_var, input_s_var)

            probs_flat = pred.view(-1)
            true_flat  = target_var.view(-1)

            #assert (masks_probs_flat >= 0. & masks_probs_flat <= 1.).all()
            loss = criterion(probs_flat, true_flat)
            losses.update(loss, input_f_var.size(0))

            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\n\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch
            }, args.model_path)

def validate(model, val_loader, criterion):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input_f, input_s, target) in enumerate(val_loader):
            input_f_var = Variable(input_f).cuda()
            input_s_var = Variable(input_s).cuda()
            target_var = Variable(target).cuda()

            output = model(input_f_var, input_s_var)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_f_var.size(0))

    return {'valid_loss': losses.avg}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir", type=str, required=True)
    ap.add_argument("-sil_s_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=True)
    ap.add_argument("-model_path_f", type=str, required=True)
    ap.add_argument("-model_path_s", type=str, required=True)
    ap.add_argument("-model_path", type=str, required=True)
    ap.add_argument("-out_dir", type=str, required=True)
    ap.add_argument("--use_pretrained",  default=True, required=False)
    ap.add_argument("--feature_extract", default=False, required=False)
    ap.add_argument('-n_epoch', default=30, type=int, metavar='N', help='number of total epochs to run')
    ap.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    ap.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    ap.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    ap.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    ap.add_argument("--batch_size", default=16, required=False)
    ap.add_argument('-num_workers', default=4, type=int, help='output dataset directory')
    args = ap.parse_args()


    input_size = 224

    train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    valid_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    target_scaler = MinMaxScaler()
    target_data = load_target(args.target_dir)
    target_scaler.fit(target_data)
    with open(os.path.join(*[args.out_dir, 'target_scaler.pkl']), 'wb') as file:
        pickle.dump(file=file, obj=target_scaler)

    # Create training and validation datasets
    train_f_dir = os.path.join(*[args.sil_f_dir, 'train'])
    train_s_dir = os.path.join(*[args.sil_s_dir, 'train'])
    valid_f_dir = os.path.join(*[args.sil_f_dir, 'valid'])
    valid_s_dir = os.path.join(*[args.sil_s_dir, 'valid'])
    train_loader = create_pair_loader(train_f_dir, train_s_dir, args.target_dir, train_transform, target_scaler)
    valid_loader = create_pair_loader(valid_f_dir, valid_s_dir, args.target_dir, valid_transform, target_scaler)

    joint_net = load_joint_net_161_train(args.model_path_f, args.model_path_s)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(joint_net.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = joint_net.to(device)
    train(train_loader, valid_loader, joint_net, criterion, optimizer, validate, args)