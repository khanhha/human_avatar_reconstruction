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
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from sklearn.externals import joblib
from pca.dense_net import JointMask
from pca.nn_util import  AverageMeter, load_target, ImgFullDataSet, load_height
from pca.nn_util import create_pair_loader, find_latest_model_path, load_pca_model, adjust_learning_rate, network_input_size
from pca.losses import SMPLLoss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.engine.engine import Engine, State, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from tensorboardX import SummaryWriter
import logging
import sys
from pca.nn_vic_model import NNModelWrapper

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

            losses.update(loss.item(), input_f.size(0))

    return {'valid_loss': losses.avg}

def create_summary_writer(args, model, data_loader, log_dir, clean_old_log = True):
    if clean_old_log:
        for path in Path(log_dir).glob('*.*'):
            os.remove(str(path))

    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    input_f, input_s, target, height = next(data_loader_iter)
    try:
        if args.model_type == 'f':
            input = (input_f, height)
        elif args.model_type == 's':
            input = (input_s, height)
        else:
            input = input_f, input_s, height
        writer.add_graph(model, input, verbose=False)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def create_loaders(args, target_trans = None, height_trans = None):
    sil_transform = transforms.Compose([transforms.ToTensor()])

    dir_sil_f = os.path.join(*[args.root_dir, 'sil_f'])
    dir_sil_s = os.path.join(*[args.root_dir, 'sil_s'])

    # Create training and validation datasets
    train_f_dir = os.path.join(*[dir_sil_f, 'train'])
    train_s_dir = os.path.join(*[dir_sil_s, 'train'])
    valid_f_dir = os.path.join(*[dir_sil_f, 'valid'])
    valid_s_dir = os.path.join(*[dir_sil_s, 'valid'])
    test_f_dir = os.path.join(*[dir_sil_f, 'test'])
    test_s_dir = os.path.join(*[dir_sil_s, 'test'])

    heights = load_height(args.height_path)

    train_ds= ImgFullDataSet(img_transform=sil_transform,
                                  dir_f=train_f_dir, dir_s=train_s_dir,
                                  dir_target=args.target_dir, id_to_heights=heights, target_transform=target_trans, height_transform=height_trans)


    valid_ds= ImgFullDataSet(img_transform=sil_transform,
                                  dir_f=valid_f_dir, dir_s=valid_s_dir,
                                  dir_target=args.target_dir, id_to_heights=heights,  target_transform=target_trans,height_transform=height_trans)


    test_ds= ImgFullDataSet(img_transform=sil_transform,
                                  dir_f=test_f_dir, dir_s=test_s_dir,
                                  dir_target=args.target_dir, id_to_heights=heights,  target_transform=target_trans,height_transform=height_trans)

    train_loader= torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader= torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader= torch.utils.data.DataLoader(test_ds,   batch_size=args.batch_size, shuffle=True, num_workers=4)

    return train_loader, valid_loader, test_loader

def create_supervised_trainer_k(model, model_type, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              output_transform=lambda y, y_pred, loss: loss.item()):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        input_f, input_s, target, height = batch #prepare_batch(batch, device=device, non_blocking=non_blocking)

        y = Variable(target).cuda()
        height_var = Variable(height).cuda()
        if model_type == 'f':
            input_f_var = Variable(input_f).cuda()
            y_pred = model(input_f_var, height_var)
        elif model_type == 's':
            input_s_var = Variable(input_s).cuda()
            y_pred = model(input_s_var, height_var)
        else:
            input_f_var = Variable(input_f).cuda()
            input_s_var = Variable(input_s).cuda()
            y_pred = model(input_f_var, input_s_var, height_var)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(y, y_pred, loss)

    return Engine(_update)

def create_supervised_evaluator_k(model, model_type, metrics={},
                                device=None, non_blocking=False,
                                output_transform=lambda y, y_pred: (y_pred, y,)):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            input_f, input_s, target, height = batch  # prepare_batch(batch, device=device, non_blocking=non_blocking)

            y = Variable(target).cuda()
            height_var = Variable(height).cuda()
            if model_type == 'f':
                input_f_var = Variable(input_f).cuda()
                y_pred = model(input_f_var, height_var)
            elif model_type == 's':
                input_s_var = Variable(input_s).cuda()
                y_pred = model(input_s_var, height_var)
            else:
                input_f_var = Variable(input_f).cuda()
                input_s_var = Variable(input_s).cuda()
                y_pred = model(input_f_var, input_s_var, height_var)

            return output_transform(y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def find_model_path(dir, hint):
    for path in Path(dir).glob('*.*'):
        if hint in str(path):
            return path
    assert False, f'no find path found with pattern {hint} found'

def load_transformations():
    #Todo: not sure if target scaling affect the target loss. don't use it now
    if args.is_scale_target:
        target_data = load_target(args.target_dir)
        target_trans = MinMaxScaler()
        target_trans.fit(target_data)
        print(f'fit target transformation. {target_trans}: min = {target_trans.data_min_}, range={target_trans.data_range_}')
    else:
        target_trans = None

    if args.is_scale_height:
        height_data = load_height(args.height_path)
        height_data = np.array([item[1] for item in height_data.items()])
        height_data = height_data.reshape(-1, 1)
        height_trans = MinMaxScaler() #the same scale as the input silhouete
        height_trans.fit(height_data)
        print(f'fit  height transform. {height_trans}: min = {height_trans.data_min_}, range= {height_trans.data_range_}')
    else:
        height_trans = None

    return target_trans, height_trans

def run(args):
    model_root_dir = os.path.join(*[args.root_dir, 'models'])
    model_dir = os.path.join(*[model_root_dir, args.model_type])
    os.makedirs(model_dir, exist_ok=True)
    if args.model_type in ['f', 's']:
        model = densenet121(pretrained=False, num_classes=args.num_classes, n_aux_input_feature=1)
    else:
        model_f_path = find_model_path(os.path.join(*[model_root_dir, 'f']), 'final_model')
        model_s_path = find_model_path(os.path.join(*[model_root_dir, 's']), 'final_model')
        assert Path(model_f_path).exists(), 'missing front model'
        assert Path(model_s_path).exists(), 'missing side model'
        model_f_wrap = NNModelWrapper.load(model_f_path)
        model_s_wrap = NNModelWrapper.load(model_s_path)
        model = JointMask(model_f=model_f_wrap.model, model_s=model_s_wrap.model, num_classes=args.num_classes)

    target_trans, height_trans = load_transformations()

    train_loader, valid_loader, test_loader = create_loaders(args, target_trans=target_trans, height_trans=height_trans)

    log_dir = os.path.join(*[args.root_dir, 'log', args.model_type])
    writer = create_summary_writer(args, model, train_loader, log_dir)

    pca_model = joblib.load(filename=args.pca_model_path)
    if pca_model.whiten:
        pca_components = np.sqrt(pca_model.explained_variance_[:, np.newaxis]) * pca_model.components_
    else:
        pca_components = pca_model.components_
    pca_components = torch.Tensor(pca_components.T).cuda()

    criterion = SMPLLoss(pca_components, use_pca_loss = args.use_pca_loss)

    optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    device = 'cuda'

    trainer = create_supervised_trainer_k(model, args.model_type,
                                        optimizer, criterion, device=device)

    evaluator = create_supervised_evaluator_k(model, args.model_type, metrics={'loss': Loss(criterion)}, device=device)

    desc = "ITERATION - loss: {:.6f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    print(f'start traininig: pca_target_transform')
    print(f'\tmodel_type = {args.model_type}')
    print(f'\tpca_target_transform: {target_trans is not None}')
    print(f'\theight_transform: {height_trans is not None}')
    print(f'\tuse pca loss: {args.use_pca_loss}')
    print(f'\tuse height input: {args.use_height}')

    @trainer.on(Events.EPOCH_STARTED)
    def training_epoch_start(engine):
        if args.use_pca_loss:
            criterion.decay_pca_weight(engine.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % args.log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(args.log_interval)

            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_nll = metrics['loss']
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        avg_nll = metrics['loss']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg loss: {:.6f}. Pca weight {:.5f}"
            .format(engine.state.epoch, avg_nll, criterion.pca_weight))

        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)

        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.COMPLETED)
    def log_end_training(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_nll = metrics['loss']

        writer.add_scalar("test/avg_loss", avg_nll, engine.state.epoch)

        pbar.n = pbar.last_print_n = 0

        #save final model
        best_model_path = find_model_path(model_dir, hint='model_best')
        core_model = torch.load(best_model_path)

        to_save = NNModelWrapper(model=core_model, model_type=args.model_type, pca_model = pca_model,
                                 use_pca_loss=args.use_pca_loss, use_height=args.use_height,
                                 pca_target_transform=target_trans, height_transform=height_trans)
        final_path = os.path.join(*[model_dir, 'final_model.pt'])
        to_save.dump(final_path)
        print(f'dump final model wrapper to {final_path}')

    def score_function(engine):
        val_loss = evaluator.state.metrics['loss']
        # Objects with highest scores will be retained.
        return -val_loss

    early_stop = EarlyStopping(patience=args.early_stop_patient, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stop)

    for path in Path(model_dir).glob('*.*'):
        os.remove(str(path))

    best_model_saver = ModelCheckpoint(model_dir,
                                       filename_prefix="model_best",
                                       score_name="val_loss",
                                       score_function=score_function,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True, save_as_state_dict=False)
    evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {'model': model})

    last_model_saver = ModelCheckpoint(model_dir,
                                       filename_prefix="checkpoint",
                                       save_interval=1,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True, save_as_state_dict=False)
    trainer.add_event_handler(Events.COMPLETED, last_model_saver, {'model': model})

    try:
        trainer.run(train_loader, max_epochs=args.n_epoch)
    except KeyboardInterrupt:
        print("Catched KeyboardInterrupt -> exit")
    except Exception as e:
        print(e)
        try:
            # open an ipython shell if possible
            import IPython
            IPython.embed()  # noqa
        except ImportError:
            print("Failed to start IPython console")

    writer.close()
    pbar.close()

import distutils
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-root_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=True)
    ap.add_argument("-model_type", type=str, required=True, choices=['f', 's', 'joint'])
    ap.add_argument("-pca_model_path", type=str, required=True)
    ap.add_argument("-height_path", type=str, required=False, default='')

    ap.add_argument('-n_epoch', default=150, type=int, metavar='N', help='number of total epochs to run')
    ap.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    ap.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    ap.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    ap.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    ap.add_argument("-batch_size", type=int, default=16, required=False)
    ap.add_argument('-num_workers', default=4, type=int, help='output dataset directory')
    ap.add_argument('-num_classes', default=50, type=int, required=False, help='output dataset directory')
    ap.add_argument('-log_interval', default=1, type=int, required=False, help='output dataset directory')
    ap.add_argument('-early_stop_patient', default=10, type=int, required=False, help='output dataset directory')
    ap.add_argument("-is_scale_target",  default=0, type=int, required=True)
    ap.add_argument("-is_scale_height",  default=0, type=int, required=True)
    ap.add_argument('-use_pca_loss',  default=0, type=int, required=True)
    ap.add_argument('-use_height',  default=1, type=int, required=True)

    args = ap.parse_args()
    assert args.use_height == True, 'only support height input currently'
    run(args)
    exit()

    # model_root_dir = os.path.join(*[args.root_dir, 'models'])
    # model_dir = os.path.join(*[model_root_dir, args.model_type])
    # os.makedirs(model_dir, exist_ok=True)
    # if args.model_type in ['f', 's']:
    #     model = densenet121(pretrained=False, num_classes=args.num_classes, n_aux_input_feature=1)
    # else:
    #     model_f_path = os.path.join(*[model_root_dir, 'f', 'model_best.pt'])
    #     model_s_path = os.path.join(*[model_root_dir, 's', 'model_best.pt'])
    #     assert Path(model_f_path).exists(), 'missing front model'
    #     assert Path(model_s_path).exists(), 'missing side model'
    #     model = load_joint_net_161_train(model_f_path, model_s_path, num_classes=args.num_classes, n_aux_input_feature=1)
    #
    # train_loader, valid_loader = create_loaders(args)
    #
    # pca_model = joblib.load(filename=args.pca_model_path)
    # if pca_model.whiten:
    #     pca_components = np.sqrt(pca_model.explained_variance_[:, np.newaxis]) * pca_model.components_
    # else:
    #     pca_components = pca_model.components_
    # pca_components = torch.Tensor(pca_components.T).cuda()
    #
    # criterion = SMPLLoss(pca_components)
    #
    # optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_ft = model.to(device)
    # train(train_loader, valid_loader, model, criterion, optimizer, validate, args, model_dir)