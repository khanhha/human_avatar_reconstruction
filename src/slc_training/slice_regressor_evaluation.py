import argparse
import pickle
import os
from slc_training.slice_train_util import load_bad_slice_names, load_slice_data_1, SlcData
import common.util as util
import matplotlib.pyplot as plt
import numpy as np

def load_contour(dir, name):
    if '.pkl' not in name:
        name = name + '.pkl'

    slc_path = os.path.join(dir, name)
    with open(slc_path, 'rb') as file:
        data = pickle.load(file)
        contour = data['cnt']

    return contour

def normalize_contour(contour, resolution):
    F = util.calc_fourier_descriptor(contour[0,:], contour[1,:], resolution)
    contour = util.reconstruct_contour_fourier(F)
    return contour

def is_valid_model(model, id):
    a = hasattr(model, 'slc_id')
    b = hasattr(model, 'slc_model_input_ids')
    c = hasattr(model, 'predict')
    d = model.slc_id == id
    return a and b and c and d

def load_models(dir_0, dir_1, model_slc_ids):
    models_0 = {}
    models_1 = {}
    for model_id in model_slc_ids:
        # load model_0
        model_path_0 = os.path.join(dir_0, f'{model_id}.pkl')
        with open(model_path_0, 'rb') as file:
            model_0 = pickle.load(file)

        assert is_valid_model(model_0, model_id) == 1

        models_0[model_0.slc_id] = model_0

        #load model 1
        model_path_1 = os.path.join(dir_1, f'{model_id}.pkl')
        with open(model_path_1, 'rb') as file:
            model_1 = pickle.load(file)

        assert is_valid_model(model_1, model_id) == 1

        models_1[model_1.slc_id] = model_1

    return models_0, models_1

def load_all_necessary_data(models_0, models_1, all_slc_dir, all_feature_dir, bad_slc_dir):

    load_slc_ids = set()
    for slc_id, model in models_0.items():
        load_slc_ids.add(slc_id)
        for id in model.slc_model_input_ids:
            load_slc_ids.add(id)

    for slc_id, model in models_1.items():
        load_slc_ids.add(slc_id)
        for id in model.slc_model_input_ids:
            load_slc_ids.add(id)

    all_slc_data = {}
    for slc_id in load_slc_ids:
        SLC_DIR = os.path.join(all_slc_dir, slc_id)
        FEATURE_DIR = os.path.join(all_feature_dir, slc_id)
        bad_fnames = load_bad_slice_names(bad_slc_dir, slc_id)
        data = load_slice_data_1(slc_id, SLC_DIR, FEATURE_DIR, bad_fnames)
        all_slc_data[slc_id] = data

    return all_slc_data


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-slc_dir",  required=True, type=str, help="root directory contains all slice directory")
    ap.add_argument("-feature_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-bad_slc_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-debug_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-model_dir_0", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-model_dir_1", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-model_ids", required=True, type=str, help="root directory contains all slice code directory")
    args = ap.parse_args()

    os.makedirs(args.debug_dir)

    model_slc_ids = args.model_ids.split(',')
    models_0, models_1 = load_models(args.model_dir_0, args.model_dir_1, model_slc_ids)

    load_slc_ids = set()
    for slc_id, model in models_0.items():
        load_slc_ids.add(slc_id)
        for id in model.slc_model_input_ids:
            load_slc_ids.add(id)

    all_slc_data = load_all_necessary_data(models_0, models_1, args.slc_dir, args.feature_dir, args.bad_slc_dir)

    for slc_id in model_slc_ids:
        debug_dir = os.path.join(args.debug_dir, slc_id)
        os.makedirs(debug_dir, exist_ok=True)
        print(f'comparing models of slice {slc_id}')
        SLC_DIR = os.path.join(args.slc_dir, slc_id)

        model_0 = models_0[slc_id]
        model_1 = models_1[slc_id]

        in_slc_data_0  = [all_slc_data[id] for id in model_0.slc_model_input_ids]
        out_slc_data_0= all_slc_data[slc_id]
        fnames_0 = SlcData.extract_shared_fnames(in_slc_data_0 + [out_slc_data_0])

        in_slc_data_1  = [all_slc_data[id] for id in model_1.slc_model_input_ids]
        out_slc_data_1 = all_slc_data[slc_id]
        fnames_1 = SlcData.extract_shared_fnames(in_slc_data_1 + [out_slc_data_1])

        fnames = fnames_0.intersection(fnames_1)

        X_0, Y_0 = SlcData.build_training_data(in_slc_data_0, out_slc_data_0, fnames)
        P_0 = model_0.predict(X_0)
        print(f'model 0 input data shape X, Y: {X_0.shape} , {Y_0.shape}')

        X_1, Y_1 = SlcData.build_training_data(in_slc_data_1, out_slc_data_1, fnames)
        P_1 = model_1.predict(X_1)
        print(f'model 1 input data shape X, Y: {X_1.shape} , {Y_1.shape}')

        for idx, name in enumerate(fnames):
            contour = normalize_contour(load_contour(SLC_DIR, name), resolution =(Y_0.shape[1] + 2) / 2)

            pred = P_0[idx,:]
            res_contour_0 = util.reconstruct_contour_fourier(pred.flatten())

            pred = P_1[idx,:]
            res_contour_1 = util.reconstruct_contour_fourier(pred.flatten())

            contour = np.concatenate([contour[:,:], contour[:,0].reshape(2,1)], axis=1)
            res_contour_0 = np.concatenate([res_contour_0[:, :], res_contour_0[:, 0].reshape(2, 1)], axis=1)
            res_contour_1 = np.concatenate([res_contour_1[:, :], res_contour_1[:, 0].reshape(2, 1)], axis=1)
            plt.clf()
            plt.subplot(121)
            plt.plot(contour[0,:], contour[1,:], '-b')
            plt.plot(res_contour_0[0, :], res_contour_0[1, :], '-r')
            plt.title(f'model_0: single input:\n {model_0.slc_model_input_ids}')
            #plt.axes().set_aspect(1.0)
            plt.subplot(122)
            plt.plot(contour[0, :], contour[1, :], '-b')
            plt.plot(res_contour_1[0, :], res_contour_1[1, :], '-r')
            plt.title(f'model_1: neighbor input:\n{model_1.slc_model_input_ids}')
            plt.savefig(os.path.join(debug_dir, f'{name}.png'))





