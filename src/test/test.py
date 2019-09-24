import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import numpy as np
from os.path import join
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from tqdm import tqdm
from common.obj_util import export_mesh, import_mesh_obj
from os.path import join
from pca.nn_util import crop_silhouette_pair_blender
import tempfile
from pca.nn_vic_model import  NNModelWrapper
from pca.dense_net import densenet121, JointMask
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset,
from torch.autograd import Variable


class ImageFolderCar(ImageFolder):
    def __getitem__(self, index):
        sample, target = super(ImageFolderCar, self).__getitem__(index)
        path, _= self.samples[index]
        return sample, target, path

def plot_silhouettes():
    dir_f = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_f_cropped/train'
    dir_s = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_s_cropped/train'
    dir_f = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/mobile_image_silhouettes/sil_f_cropped'
    dir_s = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/mobile_image_silhouettes/sil_s_cropped'
    n = 2

    paths_f = sorted([path for path in Path(dir_f).glob('*.*')])
    paths_s = sorted([path for path in Path(dir_s).glob('*.*')])

    idxs = np.random.randint(0, len(paths_f), n)
    paths_f = [paths_f[i] for i in idxs]
    paths_s = [paths_s[i] for i in idxs]

    fig, axes = plt.subplots(2, n)
    for i in range(n):
        img = cv.imread(str(paths_f[i]))
        axes[0, i].imshow(img)
    for i in range(n):
        img = cv.imread(str(paths_s[i]))
        axes[1, i].imshow(img)
    plt.show()

def load_img(path, color, bgr_color = (125,125,125)):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    mask = img > 0
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img[mask] = color
    img[np.bitwise_not(mask)] = bgr_color
    return img

def plot_diff_camera_sil():
    dir_f = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/debug/CSR0309A/sil_f_cropped/'
    dir_s = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/debug/CSR0309A/sil_s_cropped/'
    pathf_0 = join(*[dir_f, 'CSR0309A_dst_30.jpg'])
    pathf_1 = join(*[dir_f, 'CSR0309A_dst_50.jpg'])
    paths_0 = join(*[dir_s, 'CSR0309A_dst_30.jpg'])
    paths_1 = join(*[dir_s, 'CSR0309A_dst_50.jpg'])
    # pathf_0 = join(*[dir_f, 'CSR0309A_focal_len_2.6.jpg'])
    # pathf_1 = join(*[dir_f, 'CSR0309A_focal_len_4.4.jpg'])
    # paths_0 = join(*[dir_s, 'CSR0309A_focal_len_2.6.jpg'])
    # paths_1 = join(*[dir_s, 'CSR0309A_focal_len_4.4.jpg'])

    plt.rcParams['axes.facecolor'] = 'white'
    fig, axes = plt.subplots(1,2, facecolor='red')
    axes[0].imshow(load_img(pathf_0, (255,0,0)))
    axes[0].imshow(load_img(pathf_1, (0,0,255)), alpha=0.5)
    axes[1].imshow(load_img(paths_0, (255,0,0)))
    axes[1].imshow(load_img(paths_1, (0,0,255)), alpha=0.5)
    fig.set_facecolor("white")
    plt.show()

def analyze_dif_cam():
    dir_f = "/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_silhouette_diff_cam/sil_f_fc"
    dir_s = "/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_silhouette_diff_cam/sil_s_fc"
    names = sorted([path.name for path in Path(dir_f).glob('*.*')])

    size = (1024, 768)
    avg_sil_f = np.zeros(size, dtype=np.float)
    avg_sil_s = np.zeros(size, dtype=np.float)
    for idx, name in enumerate(names):
        #if idx != 0  and idx != len(names)-1:
        #      continue
        path_f = join(*[dir_f, name])
        path_s = join(*[dir_s, name])
        img_f = cv.imread(path_f, cv.IMREAD_GRAYSCALE)
        img_s = cv.imread(path_s, cv.IMREAD_GRAYSCALE)
        sil_f, sil_s = crop_silhouette_pair_blender(img_f, img_s, size=size)
        assert sil_f.max() == 255 and sil_s.max() == 255
        sil_f = sil_f.astype(np.float)/255.0
        sil_s = sil_s.astype(np.float)/255.0
        avg_sil_f += sil_f
        avg_sil_s += sil_s

    avg_sil_f /= float(len(names))
    avg_sil_s /= float(len(names))
    #avg_sil_f /= 2.0
    #avg_sil_s /= 2.0
    fig, axes = plt.subplots(1, 2, facecolor='red')
    axes[0].imshow(avg_sil_f)
    axes[1].imshow(avg_sil_s)
    fig.set_facecolor("white")
    plt.show()

def export_vic_pca_height():
    pca_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/pca_vic_coords/'
    model_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_model.jlb'
    height_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/height_syn.txt'
    pca_model = joblib.load(filename=model_path)

    paths = [path for path in Path(pca_dir).glob('*.npy')]
    heights = []
    for path in tqdm(paths):
        p = np.load(path)
        verts = pca_model.inverse_transform(p)
        verts = verts.reshape(verts.shape[0]//3, 3)
        h = verts[:,2].max() - verts[:,2].min()
        heights.append((path.stem, h))

    with open(height_path, 'wt') as file:
         file.writelines(f"{l[0]} {l[1]}\n" for l in heights)

def test_pca_max_min():
    pca_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/pca_vic_coords/'
    model_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_model.jlb'
    height_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/height_syn.txt'
    pca_model = joblib.load(filename=model_path)

    paths = [path for path in Path(pca_dir).glob('*.npy')]
    pca_vals = []
    for path in tqdm(paths):
        p = np.load(path)
        pca_vals.append(p)

    pca_vals = np.array(pca_vals)
    scale = RobustScaler()
    pca_vals_1 = scale.fit_transform(pca_vals)
    print(pca_vals.min(), pca_vals.max())
    print(pca_vals_1.min(), pca_vals_1.max())

def test_export_caesar_vic_mesh():
    vert_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/victoria_caesar/CSR0097A.pkl'
    vic_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj'
    out_mesh_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/victoria_caesar_obj/CSR0097A.obj'
    tpl_verts, tpl_faces = import_mesh_obj(vic_mesh_path)
    verts = joblib.load(vert_path)
    export_mesh(fpath=out_mesh_path, verts=verts, faces=tpl_faces)

import torch
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
from onnx_tf.pb_wrapper import TensorflowGraph
from tensorflow.core.framework import graph_pb2
import tensorflow as tf

def test_convert_pytorch_to_tensorflow():
    dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_384_256_ml_fml/models/joint/'
    model_path = join(*[dir, 'final_model.pt'])
    model_wrapper = NNModelWrapper.load(model_path)

    #TODO: replace by trained model
    # n_classes = 51
    # model_f = densenet121(pretrained=False, num_classes=n_classes, n_aux_input_feature=2)
    # model_s = densenet121(pretrained=False, num_classes=n_classes, n_aux_input_feature=2)
    # model = JointMask(model_f=model_f, model_s=model_s, num_classes=n_classes)
    # model.cuda()
    # model.eval()
    #
    # # # Export the trained model to ONNX
    # dummy_x_f = Variable(torch.randn(1, 1, 384, 256)).cuda()  # one black and white 28 x 28 picture will be the input to the model
    # dummy_x_s = Variable(torch.randn(1, 1, 384, 256)).cuda()  # one black and white 28 x 28 picture will be the input to the model
    # dummy_h_g = Variable(torch.randn(1, 2)).cuda()  # one black and white 28 x 28 picture will be the input to the model
    # torch.onnx.export(model, (dummy_x_f, dummy_x_s, dummy_h_g), f"{dir}/mnist.onnx")

    # Load the ONNX file
    model = onnx.load(f'{dir}/mnist.onnx')
    # Import the ONNX model to Tensorflow

    #strict = False. Check that issue https://github.com/onnx/onnx-tensorflow/issues/167
    tf_rep = prepare(model, strict=False)

    print('inputs:', tf_rep.inputs)

    # Output nodes from the model
    print('outputs:', tf_rep.outputs)

    #test
    sil_f = np.zeros((1,1,384,256), dtype=np.float)
    sil_s = np.zeros((1,1,384,256), dtype=np.float)
    aux   = np.zeros((1,2), dtype=np.float)
    # pred = tf_rep.run((sil_f, sil_s, aux))
    # print(pred)

    tf_rep.export_graph(f'{dir}/mnist.pt')

    # input_keys = [tf_rep.tensor_dict[key].name for key in tf_rep.inputs]
    # output_keys = [tf_rep.tensor_dict[key].name for key in tf_rep.outputs]

    # del model
    # del tf_rep
    #
    # graph = tf.Graph()
    # with graph.as_default():
    #     with tf.gfile.FastGFile(f'{dir}/mnist.pt', 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #
    #         elm_keys = input_keys + output_keys
    #         elms = tf.import_graph_def(graph_def, return_elements=elm_keys)
    #         tf_graph_inputs = elms[:len(input_keys)]
    #         tf_graph_outputs = elms[len(input_keys):]
    #
    # with graph.as_default():
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #
    #         feed_dict = {tf_graph_inputs[0]: sil_f, tf_graph_inputs[1]: sil_s, tf_graph_inputs[2]: aux}
    #
    #         output = sess.run(tf_graph_outputs, feed_dict=feed_dict)
    #
    #         print(output)
    # exit()

    graph_proto =tf_rep.graph.as_graph_def()
    graph_str = graph_proto.SerializeToString()

    input_keys = [tf_rep.tensor_dict[key].name for key in tf_rep.inputs]
    output_keys = [tf_rep.tensor_dict[key].name for key in tf_rep.outputs]
    input_shape = tf_rep.tensor_dict[tf_rep.inputs[0]].get_shape().as_list()
    print(input_shape)
    print(input_keys)
    print(output_keys)
    to_save = {}
    to_save['tf_graph_str'] = graph_str
    to_save['tf_graph_input_keys'] = input_keys
    to_save['tf_graph_output_keys'] = output_keys
    to_save['image_input_shape'] = tuple(input_shape[2:])
    to_save['model_type'] = model_wrapper.model_type
    to_save['pca_model'] = model_wrapper.pca_model
    to_save['pca_target_transform'] = model_wrapper.pca_target_transform
    to_save['aux_input_transform'] = model_wrapper.height_transform

    out_dir ='/home/khanhhh/data_1/projects/Oh/data/3d_human/deploy_models/'
    joblib.dump(value=to_save, filename=f'{out_dir}/shape_model.jlb')

if __name__ == '__main__':
    print(dir())
    #test_convert_pytorch_to_tensorflow()

    #test_pca_max_min()
    #test_export_caesar_vic_mesh()
    #analyze_dif_cam()