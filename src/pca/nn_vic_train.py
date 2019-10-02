import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pca.dense_net import densenet121
import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from pca.dense_net import JointMask
from pca.nn_util import  load_target, ImgFullDataSet, ImgFullDataSetPoseVariants, load_height
from ignite.metrics import Accuracy, Loss
from ignite.engine.engine import Engine, State, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from tensorboardX import SummaryWriter
from pca.nn_vic_model import NNModelWrapper, NNHmModel, NNHmJointModel
from pca.pca_vic_model import PcaModel
from pca.losses import SMPLLoss
import matplotlib.pyplot as plt
import PIL
from common.obj_util import import_mesh_tex_obj, export_mesh
from scipy.misc import imread, imsave
import tempfile
import cv2 as cv
import shutil

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
        #TODO: writer doesn't work with tuple of 3 inputs
        #writer.add_graph(model, input, verbose=False)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def create_loaders(args, img_transform = None, target_trans = None, height_trans = None, n_pose_variant = 0):
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

    if n_pose_variant == 0:
        train_ds= ImgFullDataSet(img_transform=img_transform,
                                      dir_f=train_f_dir, dir_s=train_s_dir,
                                      dir_target=args.target_dir, id_to_heights=heights, target_transform=target_trans, height_transform=height_trans, use_input_gender=args.use_gender)


        valid_ds= ImgFullDataSet(img_transform=img_transform,
                                      dir_f=valid_f_dir, dir_s=valid_s_dir,
                                      dir_target=args.target_dir, id_to_heights=heights,  target_transform=target_trans,height_transform=height_trans, use_input_gender=args.use_gender)


        test_ds= ImgFullDataSet(img_transform=img_transform,
                                      dir_f=test_f_dir, dir_s=test_s_dir,
                                      dir_target=args.target_dir, id_to_heights=heights,  target_transform=target_trans,height_transform=height_trans, use_input_gender=args.use_gender)
    else:

        train_ds= ImgFullDataSetPoseVariants(img_transform=img_transform,
                                 dir_f=train_f_dir, dir_s=train_s_dir,
                                 dir_target=args.target_dir, id_to_heights=heights, target_transform=target_trans, height_transform=height_trans, use_input_gender=args.use_gender,
                                             n_pose_variant=n_pose_variant, shuffle_front_side_pairs=True)

        valid_ds= ImgFullDataSetPoseVariants(img_transform=img_transform,
                                 dir_f=valid_f_dir, dir_s=valid_s_dir,
                                 dir_target=args.target_dir, id_to_heights=heights,  target_transform=target_trans,height_transform=height_trans, use_input_gender=args.use_gender,
                                             n_pose_variant=n_pose_variant, shuffle_front_side_pairs=True)

        test_ds= ImgFullDataSetPoseVariants(img_transform=img_transform,
                                dir_f=test_f_dir, dir_s=test_s_dir,
                                dir_target=args.target_dir, id_to_heights=heights,  target_transform=target_trans,height_transform=height_trans, use_input_gender=args.use_gender,
                                            n_pose_variant=n_pose_variant, shuffle_front_side_pairs=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=args.batch_size, shuffle=True, num_workers=4)

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
    return None

def generate_weight(num_classes):
    weight = torch.ones(num_classes)
    #gender
    weight[0] = 100
    weight[1] = 100

def load_transformations():
    print('calculating PCA target and height transformations')
    use_min_max_scale = False
    if args.is_scale_target:
        target_data = load_target(args.target_dir)
       # T = np.array(target_data)
       # for i in range(T.shape[1]):
       #     X = T[:,i]
       #     plt.hist(X, bins = 200)
       #     plt.title(f'hist {i}')
       #     plt.show()

        if use_min_max_scale:
            target_trans = MinMaxScaler()
            target_trans.fit(target_data)
            print(f'fit min-max target transformation. {target_trans}: \n range={target_trans.data_range_}') #min = {target_trans.data_min_}
        else:
            #target_trans = StandardScaler()
            #target_trans.fit(target_data)
            #print(f'fit standard scaler target transform: {target_trans}: \n mean = {target_trans.mean_}, \n scale = {target_trans.scale_}')
            target_trans = RobustScaler()
            target_trans.fit(target_data)
            print(f'fit robust scaler target transform: {target_trans}')
        test = target_trans.transform(target_data)
        print(f'min before transform = {np.min(target_data, axis=0)}')
        print(f'max before transform = {np.max(target_data, axis=0)}')
        print(f'min after transform = {np.min(test, axis=0)}')
        print(f'max after transform = {np.max(test, axis=0)}')

       # T = np.array(test)
       # for i in range(T.shape[1]):
       #     X = T[:,i]
       #     plt.hist(X, bins = 200)
       #     plt.title(f'after transformed hist {i}')
       #     plt.show()
    else:
        target_trans = None

    if args.is_scale_height:
        height_data = load_height(args.height_path)
        height_data = np.array([item[1] for item in height_data.items()])
        height_data = height_data.reshape(-1, 1)
        if use_min_max_scale:
            height_trans = MinMaxScaler() #the same scale as the input silhouete
            height_trans.fit(height_data)
            print(f'fit  min-max scaler height transform. {height_trans}: min = {height_trans.data_min_}, range= {height_trans.data_range_}')
        else:
            #height_trans = StandardScaler()
            height_trans = RobustScaler()
            height_trans.fit(height_data)
            #print(f'fit standard scaler height transform. {height_trans}')
            print(f'fit robust scaler height transform: {height_trans}')

        test = height_trans.transform(height_data)
        print(f'min before transform = {np.min(height_data, axis=0)}')
        print(f'max before transform = {np.max(height_data, axis=0)}')
        print(f'min after transform = {np.min(test, axis=0)}')
        print(f'max after transform = {np.max(test, axis=0)}')
    else:
        height_trans = None

    #exit()
    return target_trans, height_trans

def load_model(args, model_root_dir):
    use_old_architecture = False
    #train the front and side model first
    if args.model_type in ['f', 's']:
        n_aux_input_feature = 1 #height
        if args.use_gender:
            n_aux_input_feature = 2 #gender
        if use_old_architecture:
            model = densenet121(pretrained=False, num_classes=args.num_classes, n_aux_input_feature=n_aux_input_feature)
        else:
            model = NNHmModel(num_classes=args.num_classes, n_aux_input_feature=n_aux_input_feature, encoder_type=args.encoder_type)
    else:
        #find the path to the front and side models
        model_f_path = find_model_path(os.path.join(*[model_root_dir, 'f']), 'final_model')
        model_s_path = find_model_path(os.path.join(*[model_root_dir, 's']), 'final_model')
        assert Path(model_f_path).exists(), 'missing front model'
        assert Path(model_s_path).exists(), 'missing side model'
        model_f_wrap = NNModelWrapper.load(model_f_path)
        model_s_wrap = NNModelWrapper.load(model_s_path)
        print(f'initialize joint model from model_f and model_s')
        print(f'\tmodel_f: {model_f_path}')
        print(f'\tmodel_s: {model_s_path}')
        print('\n')
        #create  joint model from the front and side weights
        if use_old_architecture:
            model = JointMask(model_f=model_f_wrap.model, model_s=model_s_wrap.model, num_classes=args.num_classes)
        else:
            model = NNHmJointModel(model_f=model_f_wrap.model, model_s=model_s_wrap.model, num_classes=args.num_classes)

    return model

def build_image_transform(args):
    in_img_size = (224, 224)
    use_input_color = args.is_color
    if use_input_color:
        print('image_transfromation: use body part color training image ')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
        img_train_transform = transforms.Compose([
            #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            #transforms.RandomAffine(0, translate=(0.05, 0.0), resample=PIL.Image.BILINEAR),
            #transforms.RandomRotation(degrees=2),
            transforms.Resize(in_img_size),
            transforms.ToTensor()])

        img_test_transform = transforms.Compose([
            transforms.Resize(in_img_size),
            transforms.ToTensor()])
    else:
        print('image_transformation: use color training image')
        img_train_transform = transforms.Compose([transforms.Resize(in_img_size), transforms.Grayscale(3), transforms.ToTensor()])
        img_test_transform = img_train_transform

    return img_train_transform, img_test_transform

def load_test_data_paths(data_file):
    #load front and side images
    fpaths = []
    spaths = []
    heights = []
    genders = []
    with open(data_file, 'rt') as file:
        dir = Path(data_file).parent

        for idx, l in enumerate(file.readlines()):
            if l[0] == '#':
                continue
            l = l.replace('\n','')
            comps = l.split(' ')
            if len(comps) < 2:
                print("ignore line: ", comps, f'len(comps) = {len(comps)}')
                continue

            fpath = Path(os.path.join(*[dir, comps[0]]))
            spath  = Path(os.path.join(*[dir, comps[1]]))
            if len(comps) >= 4:
                height = float(comps[2])
                gender = float(comps[3])
            elif len(comps) == 2:
                #default height and gender
                height = 1.6
                gender = 0.0
            else:
                print("ignore line: ", comps, f'len(comps) = {len(comps)}')
                continue

            fpaths.append(fpath)
            spaths.append(spath)
            heights.append(height)
            genders.append(gender)

    return fpaths, spaths, heights, genders


from mpl_toolkits.mplot3d import Axes3D
def project_silhouette(verts, triangles):
    half_height = 1.0

    plt.clf()
    ax = plt.gca(projection='3d')
    center = 0.5*(verts.max(0) + verts.min(0))
    verts = verts - center

    # draw the mesh. set fixed color to avoid random color
    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=triangles, color='grey', linewidth=0, antialiased=False)

    #TODO: trick to force matplotlib to ensure equal scaling across axes
    # Create cubic bounding box to simulate equal aspect ratio
    # link: https://python-decompiler.com/article/2012-12/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    X, Y, Z = verts[:,0], verts[:,1], verts[:,2]
    xmax, xmin = X.max(), X.min()
    ymax, ymin = Y.max(), Y.min()
    zmax, zmin = Z.max(), Z.min()
    max_range = np.array([xmax- xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.axis('off')
    plt.grid(b=None)

    #TODO: fixed image range extraction for dpi = 300 to crop the silhouette region in image.
    # if you change dpi to another different number, you have to calibrate this fixed range.
    dpi = 300
    yrange_dpi = (700-350,700+350)
    xrange_dpi = (1000-250,1000+250)
    #trick to get image data from matplotlib: save render outptu to a temp file and then read back
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_f = f'{tmp_dir}/img_front.png'
        ax.view_init(0, -90)
        plt.savefig(path_f, dpi = dpi)
        img_f = imread(path_f, mode='RGB')
        img_f = img_f[yrange_dpi[0]:yrange_dpi[1], xrange_dpi[0]:xrange_dpi[1] ,:]

        path_s = f'{tmp_dir}/img_side.png'
        ax.view_init(0, 0)
        plt.savefig(path_s, dpi=dpi)
        img_s = imread(path_s, mode='RGB')
        img_s = img_s[yrange_dpi[0]:yrange_dpi[1], xrange_dpi[0]:xrange_dpi[1], :]

    return img_f, img_s

def resize_height(img_src, img_target):
    tmax = img_target.shape[0]
    smax = img_src.shape[0]
    scale = tmax/smax
    dsize = (int(img_src.shape[1]*scale), img_target.shape[0])
    img_src_1 = cv.resize(img_src, dsize=dsize)
    return img_src_1

def resize_width(img_src, img_target):
    tmax = img_target.shape[1]
    smax = img_src.shape[1]
    scale = tmax/smax
    dsize = (img_target.shape[1], int(scale*img_src.shape[0]))
    img_src_1 = cv.resize(img_src, dsize=dsize)
    return img_src_1

def viz_prediction_dynamics(board_writer, model, model_type, pca_model_wrapper,
                            img_transform, target_transform, aux_input_transform, in_data_file,
                            vic_mesh_path, out_dir, iteration):


    mesh = import_mesh_tex_obj(vic_mesh_path)
    faces = mesh['f']

    fpaths, spaths, heights, genders = load_test_data_paths(in_data_file)

    tab_cnt = 0
    for fpath, spath, height, gender in zip(fpaths, spaths, heights, genders):
        img_f_org = PIL.Image.open(str(fpath))
        img_s_org = PIL.Image.open(str(spath))

        img_f = img_transform(img_f_org).unsqueeze(0)
        img_s = img_transform(img_s_org).unsqueeze(0)

        aux = np.array([height])[np.newaxis, :]
        aux = aux_input_transform.transform(aux)
        aux = np.array([aux[0,0], gender])[np.newaxis, :]
        aux = aux.astype(np.float32)

        with torch.no_grad():
            aux_var = Variable(torch.from_numpy(aux)).cuda()
            if model_type == 'f':
                input_f_var = Variable(img_f).cuda()
                pred = model(input_f_var, aux_var)
            elif model_type == 's':
                input_s_var = Variable(img_s).cuda()
                pred = model(input_s_var, aux_var)
            elif model_type == 'joint':
                input_f_var = Variable(img_f).cuda()
                input_s_var = Variable(img_s).cuda()
                pred = model(input_f_var, input_s_var, aux_var)
            pred = pred.data.cpu().numpy()

            #pass prediction values to PCA wrapper to reconstruct mesh vertices.
            pred = target_transform.inverse_transform(pred.reshape(1,-1))
            verts = pca_model_wrapper.inverse_transform(pred)
            verts = verts[0]
            verts = verts.reshape(verts.shape[0] // 3, 3)
            export_mesh(f'{out_dir}/{fpath.stem}.obj', verts=verts, faces=faces)

            #glue all images into a big one for visualization
            img_f_pred, img_s_pred = project_silhouette(verts, faces)
            img_f_org = resize_height(np.array(img_f_org), img_f_pred)
            img_s_org = resize_height(np.array(img_s_org), img_s_pred)

            img_f_pair = np.concatenate((img_f_org, img_f_pred),axis=1)
            img_s_pair = np.concatenate((img_s_org, img_s_pred),axis=1)

            img_s_pair = resize_width(img_s_pair, img_f_pair)
            img_full = np.concatenate((img_f_pair, img_s_pair), axis=0)

            imsave(f'{out_dir}/{fpath.stem}.png', img_full)
            board_writer.add_image(f"{tab_cnt//3}/{fpath.stem}", np.transpose(img_full, (2,0,1)), global_step=iteration)

            tab_cnt += 1

def run(args):
    model_root_dir = os.path.join(*[args.root_dir, 'models'])
    model_dir = os.path.join(*[model_root_dir, args.model_type])
    os.makedirs(model_dir, exist_ok=True)

    img_transform, img_test_transform = build_image_transform(args)

    pca_model = PcaModel.load(args.pca_model_path)

    model = load_model(args, model_root_dir)

    target_trans, aux_input_trans = load_transformations()

    print('create data loaders: train, valid, test')
    train_loader, valid_loader, test_loader = create_loaders(args, img_transform=img_transform, target_trans=target_trans, height_trans=aux_input_trans, n_pose_variant=args.n_pose_variant)

    #clean log
    log_dir = os.path.join(*[args.root_dir, 'log', args.model_type])
    shutil.rmtree(log_dir, ignore_errors=True); os.makedirs(log_dir, exist_ok=True)
    writer = create_summary_writer(args, model, train_loader, log_dir)

    #optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)
    device = 'cuda'

    # if if mesh_loss_vert_idxs_path is available,
    # we calculate the mesh vertex loss on a subset of vertices rather than the whole vertex list.
    mesh_loss_vert_idxs = None
    if args.mesh_loss_vert_idxs_path != '':
        mesh_loss_vert_idxs = np.load(args.mesh_loss_vert_idxs_path)
        print(f'apply mesh loss on a subset of vertices: N_subset = {mesh_loss_vert_idxs.shape}')
    else:
        print(f'apply mesh loss on the whole mesh vertex list')

    pca_comps_famle = pca_model.model_female.components_
    pca_comps_male = pca_model.model_male.components_
    loss_fn = SMPLLoss(num_classes=args.num_classes,
                       pca_comps_male= pca_comps_male, pca_comps_female= pca_comps_famle,
                       use_weighted_mse_loss=True, mesh_vert_idxs=mesh_loss_vert_idxs,  pca_start_idx=1)

    trainer = create_supervised_trainer_k(model, args.model_type,
                                        optimizer, loss_fn, device=device)

    train_evaluator = create_supervised_evaluator_k(model, args.model_type, metrics={'loss': Loss(loss_fn)}, device=device)
    valid_evaluator = create_supervised_evaluator_k(model, args.model_type, metrics={'loss': Loss(loss_fn)}, device=device)

    desc = "ITERATION - loss: {:.6f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    print(f'start traininig: pca_target_transform')
    print(f'\tmodel_type = {args.model_type}')
    print(f'\tpca_target_transform: {target_trans is not None}')
    print(f'\theight_transform: {aux_input_trans is not None}')
    print(f'\tuse height input: {args.use_height}')
    print(f'\tuse gender input: {args.use_gender}')
    print(f'\tnumber of pose variant: {args.n_pose_variant}')
    print(f'\tcolor image: {args.is_color}')
    print(f'\tmornitor with sparse vertex set: {len(args.mesh_loss_vert_idxs_path) != 0}')

    #save result before the model is trained
    if args.in_test_file:
        print('Export intermediate result on the sample test: epoch 0 ')
        vic_mesh_path = Path(os.path.join(args.root_dir, "vic_template_triangle_mesh.obj"))
        if vic_mesh_path.exists():
            cur_out_dir = f'{log_dir}/log_mesh/epoch_{0}/'
            os.makedirs(cur_out_dir, exist_ok=True)
            viz_prediction_dynamics(writer, model, args.model_type, pca_model,
                                    img_test_transform, target_trans, aux_input_trans,
                                    args.in_test_file, vic_mesh_path, cur_out_dir, 0)
        else:
            print('vic mesh path does not exist. Unable to export test result during training. mesh_path = ',
                  vic_mesh_path)

    def save_best_model_wrapper():
        # save final model
        best_model_path = find_model_path(model_dir, hint='model_best')
        if best_model_path is not None:
            core_model = torch.load(best_model_path)

            to_save = NNModelWrapper(model=core_model, model_type=args.model_type, pca_model=pca_model,
                                     use_pca_loss=False, use_height=args.use_height,
                                     img_input_transform = img_test_transform,
                                     pca_target_transform = target_trans, height_transform=aux_input_trans)
            final_path = os.path.join(*[model_dir, 'final_model.pt'])
            to_save.dump(final_path)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % args.log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(args.log_interval)

            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):

        mesh_loss_weight = 0.0
        if hasattr(loss_fn, 'loss_update_per_epoch'):
            loss_fn.loss_update_per_epoch(epoch=engine.state.epoch)
        if hasattr(loss_fn, 'mesh_loss_weight'):
            mesh_loss_weight = loss_fn.mesh_loss_weight

        # evaluate average training loss
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        train_loss = metrics['loss']
        writer.add_scalar("training/avg_loss", train_loss, engine.state.epoch)

        # evaluate valid loss
        valid_evaluator.run(valid_loader)
        metrics = valid_evaluator.state.metrics
        valid_loss = metrics['loss']
        tqdm.write(
            "Validation Results - Epoch: {}, train loss: {:.6f}, valid loss: {:.6f}. mesh_loss_weight = {:.4f}"
            .format(engine.state.epoch, train_loss, valid_loss, mesh_loss_weight))

        writer.add_scalar("valdation/avg_loss", valid_loss, engine.state.epoch)

        # save best wrapper, after the best model is already dumped by BestModeSaver
        save_best_model_wrapper()

        # decay learning rate
        scheduler.step()

        pbar.n = pbar.last_print_n = 0

        #export the temporary test result during training
        log_step = 4
        if args.in_test_file and (engine.state.epoch >=log_step and engine.state.epoch%log_step == 0):
            print('Export intermediate result on the sample test: epoch : ', engine.state.epoch)
            vic_mesh_path = Path(os.path.join(args.root_dir, "vic_template_triangle_mesh.obj"))
            if vic_mesh_path.exists():
                cur_out_dir = f'{log_dir}/log_mesh/epoch_{engine.state.epoch}/'
                os.makedirs(cur_out_dir, exist_ok=True)
                viz_prediction_dynamics(writer, model, args.model_type, pca_model,
                                        img_test_transform, target_trans, aux_input_trans,
                                        args.in_test_file, vic_mesh_path, cur_out_dir, engine.state.epoch)
            else:
                print('vic mesh path does not exist. Unable to export test result during training. mesh_path = ',
                      vic_mesh_path)

    @trainer.on(Events.COMPLETED)
    def log_end_training(engine):
        train_evaluator.run(test_loader)
        metrics = train_evaluator.state.metrics
        test_loss = metrics['loss']

        writer.add_scalar("test/avg_loss", test_loss, engine.state.epoch)

        pbar.n = pbar.last_print_n = 0

    def score_function(engine):
        val_loss = valid_evaluator.state.metrics['loss']
        # Objects with highest scores will be retained.
        return -val_loss

    early_stop = EarlyStopping(patience=args.early_stop_patient, score_function=score_function, trainer=trainer)
    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stop)

    for path in Path(model_dir).glob('*.*'):
        os.remove(str(path))

    best_model_saver = ModelCheckpoint(model_dir,
                                       filename_prefix="model_best",
                                       score_name="val_loss",
                                       score_function=score_function,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True, save_as_state_dict=False)
    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_model_saver, {'model': model})

    last_model_saver = ModelCheckpoint(model_dir,
                                       filename_prefix="checkpoint",
                                       save_interval=1,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True, save_as_state_dict=False)
    trainer.add_event_handler(Events.COMPLETED, last_model_saver, {'model': model})

    trainer.run(train_loader, max_epochs=args.n_epoch)

    writer.close()
    pbar.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-root_dir", type=str, required=True)
    ap.add_argument("-target_dir", type=str, required=True)
    ap.add_argument("-model_type", type=str, required=True, choices=['f', 's', 'joint'])
    ap.add_argument("-pca_model_path", type=str, required=True)
    ap.add_argument("-height_path", type=str, required=False, default='')

    ap.add_argument('-n_epoch', default=30, type=int, metavar='N', help='number of total epochs to run')
    ap.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    ap.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    ap.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    ap.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    ap.add_argument("-batch_size", type=int, default=16, required=False)
    ap.add_argument('-num_workers', default=4, type=int, help='output dataset directory')
    ap.add_argument('-num_classes', default=50, type=int, required=False, help='output dataset directory')
    ap.add_argument('-encoder_type', default='vgg16_bn', type=str, choices=['resnet18', 'vgg16_bn', 'densenet'])
    ap.add_argument('-log_interval', default=1, type=int, required=False, help='output dataset directory')
    ap.add_argument('-early_stop_patient', default=10, type=int, required=False, help='output dataset directory')
    ap.add_argument("-is_scale_target",  default=0, type=int, required=True)
    ap.add_argument("-is_scale_height",  default=0, type=int, required=True)
    ap.add_argument('-use_height',  default=1, type=int, required=True)
    ap.add_argument('-use_gender',  default=1, type=int, required=True)
    ap.add_argument('-n_pose_variant', default=0, type=int, required=False, help='number of pose varaint per subject. man0_pose0, ..., man0_pose29')
    ap.add_argument('-mesh_loss_vert_idxs_path', default='', type=str, required=False, help='normally, the mesh loss is calculated over the whole mesh vertices. if this field is not empty, '
                                                                                            'the mesh loss will be calculated on a subset of vertex. this must be a *.npy file')

    ap.add_argument("-is_color", action='store_true',help="train with color or binary image?")
    ap.add_argument("-in_test_file", default='', type=str, required=False, help="data.txt file that list the front/side/height/gender to predict mesh during training")
    #ap.add_argument("-vic_mesh_path", default='', type=str, required=False, help="template victoria mesh. this must be provided when you want to ouput mesh during training")
    args = ap.parse_args()
    assert args.use_height == True, 'only support height input currently'

    np.random.seed(0)
    torch.random.manual_seed(0)
    run(args)
