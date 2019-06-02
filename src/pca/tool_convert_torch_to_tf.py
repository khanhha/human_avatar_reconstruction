import torch
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
from pca.nn_vic_model import  NNModelWrapper
import os
import numpy as np
from sklearn.externals import joblib
from common.obj_util import import_mesh_obj
import tempfile

def convert(model_path, out_path, vic_mesh_path):
    model_wrapper = NNModelWrapper.load(model_path)
    #for testing, in case there's not pre-trained model
    # n_classes = 51
    # model_f = densenet121(pretrained=False, num_classes=n_classes, n_aux_input_feature=2)
    # model_s = densenet121(pretrained=False, num_classes=n_classes, n_aux_input_feature=2)
    # model = JointMask(model_f=model_f, model_s=model_s, num_classes=n_classes)
    model_torch = model_wrapper.model
    model_torch.cuda()
    model_torch.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Export the trained model to ONNX
        dummy_x_f = Variable(torch.randn(1, 1, 384, 256)).cuda()  # one black and white 28 x 28 picture will be the input to the model
        dummy_x_s = Variable(torch.randn(1, 1, 384, 256)).cuda()  # one black and white 28 x 28 picture will be the input to the model
        dummy_h_g = Variable(torch.randn(1, 2)).cuda()  # one black and white 28 x 28 picture will be the input to the model
        torch.onnx.export(model_torch, (dummy_x_f, dummy_x_s, dummy_h_g), f"{tmp_dir}/mnist.onnx")

        # Load the ONNX file
        model = onnx.load(f'{tmp_dir}/mnist.onnx')

    #Import the ONNX model to Tensorflow
    #strict = False. Check that issue https://github.com/onnx/onnx-tensorflow/issues/167
    tf_rep = prepare(model, strict=False)
    #print('inputs:', tf_rep.inputs)
    #print('outputs:', tf_rep.outputs)

    graph_proto = tf_rep.graph.as_graph_def()
    graph_str = graph_proto.SerializeToString()

    input_keys = [tf_rep.tensor_dict[key].name for key in tf_rep.inputs]
    output_keys = [tf_rep.tensor_dict[key].name for key in tf_rep.outputs]
    input_shape = tf_rep.tensor_dict[tf_rep.inputs[0]].get_shape().as_list()

    tpl_verts, tpl_faces = import_mesh_obj(vic_mesh_path)

    to_save = {}
    to_save['tf_graph_str'] = graph_str
    to_save['tf_graph_input_keys'] = input_keys
    to_save['tf_graph_output_keys'] = output_keys
    to_save['image_input_shape'] = tuple(input_shape[2:])
    to_save['model_type'] = model_wrapper.model_type
    to_save['pca_model'] = model_wrapper.pca_model
    to_save['pca_target_transform'] = model_wrapper.pca_target_transform
    to_save['aux_input_transform'] = model_wrapper.height_transform
    to_save['vic_caesar_faces'] = tpl_faces

    joblib.dump(value=to_save, filename=out_path)
    print(f'dump the output model to {out_path}')

import argparse
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-in_model_path", type=str, help='the path to the file: DATA_DIR/models/joint/final_model.pt')
    ap.add_argument("-out_model_path", type=str, help='the path to export model to: shape_model.jlb')
    ap.add_argument("-vic_mesh_path", type=str, help='the path to victoria template mesh: victoria_caesar_template.obj')
    args = ap.parse_args()

    model_path = args.in_model_path
    out_path = args.out_model_path
    convert(model_path, out_path, args.vic_mesh_path)