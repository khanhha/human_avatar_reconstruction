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

        assert is_valid_model(model_0['model'], model_id) == 1

        models_0[model_id] = model_0

        #load model 1
        model_path_1 = os.path.join(dir_1, f'{model_id}.pkl')
        with open(model_path_1, 'rb') as file:
            model_1 = pickle.load(file)

        assert is_valid_model(model_1['model'], model_id) == 1

        models_1[model_id] = model_1

    return models_0, models_1

def load_all_slice_data(models_0, models_1, all_slc_dir, all_feature_dir, bad_slc_dir):

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

def load_shared_file_names(models_0, models_1, all_slc_data):

    shared_fnames = set()

    for id, model in  models_0.items():
        in_slc_data = [all_slc_data[id] for id in model.slc_model_input_ids]
        out_slc_data = all_slc_data[id]
        if len(shared_fnames) == 0:
            shared_fnames = SlcData.extract_shared_fnames(in_slc_data + [out_slc_data])
        else:
            shared_fnames = shared_fnames.intersection(SlcData.extract_shared_fnames(in_slc_data + [out_slc_data]))

    for id, model in  models_1.items():
        in_slc_data = [all_slc_data[id] for id in model.slc_model_input_ids]
        out_slc_data = all_slc_data[id]
        shared_fnames = shared_fnames.intersection(SlcData.extract_shared_fnames(in_slc_data + [out_slc_data]))

    return list(shared_fnames)

def load_test_names_from_all_models(models_0_data, models_1_data):
    test_names = set()
    for id, model_data in models_0_data.items():
        if len(test_names) == 0:
            test_names = set(model_data['test_fnames'])
        else:
            test_names = test_names.intersection(set(model_data['test_fnames']))

    for id, model_data in models_1_data.items():
        test_names = test_names.intersection(set(model_data['test_fnames']))

    return list(test_names)

def all_used_ids(models_0, models_1):
    load_slc_ids = set()

    for slc_id, model in models_0.items():
        load_slc_ids.add(slc_id)
        for id in model.slc_model_input_ids:
            load_slc_ids.add(id)

    for slc_id, model in models_1.items():
        load_slc_ids.add(slc_id)
        for id in model.slc_model_input_ids:
            load_slc_ids.add(id)

    return load_slc_ids

def align_ids(ids):
    out = None
    for i in range(0,len(ids),2):
        id_i = ids[i]
        if i + 1 < len(ids):
            id_i1 = ids[i+1]
        else:
            id_i1 = ''

        if out == None:
            out = f'{id_i}, {id_i1}\n'
        else:
            out = out + f', {id_i}, {id_i1}\n'

    return out

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

    #debug_dir = os.path.join(args.debug_dir, 'Crotch_to_Hip')
    os.makedirs(args.debug_dir, exist_ok=True)

    model_slc_ids = args.model_ids.split(',')
    models_0_data, models_1_data = load_models(args.model_dir_0, args.model_dir_1, model_slc_ids)
    models_0, models_1 = {}, {}
    for id, model_data in models_0_data.items():
        models_0[id] = model_data['model']
    for id, model_data in models_1_data.items():
        models_1[id] = model_data['model']

    load_slc_ids = all_used_ids(models_0, models_1)

    all_slc_data = load_all_slice_data(models_0, models_1, args.slc_dir, args.feature_dir, args.bad_slc_dir)

    #collect file names shared by all slice data
    print('collect file names shared by all slice')
    #shared_fnames = load_shared_file_names(models_0, models_1, all_slc_data)
    shared_fnames = load_test_names_from_all_models(models_0_data, models_1_data)
    print(f'n test names = {len(shared_fnames)}')

    #predict result by all models
    print('calculate prediction by all models')
    models_0_preds = {}
    for id, model in models_0.items():
        in_slc_data = [all_slc_data[id] for id in model.slc_model_input_ids]
        out_slc_data = all_slc_data[id]
        X, _ = SlcData.build_training_data(in_slc_data, out_slc_data, shared_fnames)
        P = model.predict(X)
        models_0_preds[id] = P

    models_1_preds = {}
    for id, model in models_1.items():
        in_slc_data = [all_slc_data[id] for id in model.slc_model_input_ids]
        out_slc_data = all_slc_data[id]
        X, _ = SlcData.build_training_data(in_slc_data, out_slc_data, shared_fnames)
        P = model.predict(X)
        models_1_preds[id] = P

    fontsize = 2
    for idx, name in enumerate(shared_fnames):
        contours = {}
        res_contours_0 = {}
        res_contours_1 = {}

        for slc_id in model_slc_ids:
            SLC_DIR = os.path.join(args.slc_dir, slc_id)
            contours[slc_id] = normalize_contour(load_contour(SLC_DIR, name), resolution = 20)

            pred = models_0_preds[slc_id][idx,:]
            res_contours_0[slc_id] = util.reconstruct_contour_fourier(pred.flatten())

            pred = models_1_preds[slc_id][idx,:]
            res_contours_1[slc_id] = util.reconstruct_contour_fourier(pred.flatten())

        n_slc = len(model_slc_ids)
        fig, axs = plt.subplots(2, n_slc)
        axs = axs.reshape(2,n_slc)
        for i, slc_id in enumerate(model_slc_ids):
            contour = contours[slc_id]
            res_contour_0 = res_contours_0[slc_id]
            res_contour_1 = res_contours_1[slc_id]

            contour = np.concatenate([contour[:,:], contour[:,0].reshape(2,1)], axis=1)
            res_contour_0 = np.concatenate([res_contour_0[:, :], res_contour_0[:, 0].reshape(2, 1)], axis=1)
            res_contour_1 = np.concatenate([res_contour_1[:, :], res_contour_1[:, 0].reshape(2, 1)], axis=1)

            inputs = models_0[slc_id].slc_model_input_ids
            ax = axs[0,i]
            ax.set_aspect(1.0)
            ax.plot(contour[0,:], contour[1,:], '-b')
            ax.plot(res_contour_0[0,:], res_contour_0[1,:], '-r')
            ax.set_title(f'model name = {slc_id}\n inputs = \n{align_ids(inputs)}', fontsize=fontsize, loc='left')
            ax.set_axis_off()

            inputs = models_1[slc_id].slc_model_input_ids
            ax = axs[1,i]
            ax.set_aspect(1.0)
            ax.plot(contour[0,:], contour[1,:], '-b')
            ax.plot(res_contour_1[0,:], res_contour_1[1,:], '-r')
            ax.set_title(f'model name = {slc_id}\n inputs = \n{align_ids(inputs)}', fontsize=fontsize, loc='left')
            ax.set_axis_off()

        title = f'{name}\n 1st row: single model. 2st row: neighbor model. \n green: ground truth. red: prediction'
        fig.suptitle(title, fontsize=fontsize+2)
        #plt.show()
        plt.savefig(os.path.join(args.debug_dir, f'{name}.png'), dpi=300)
        plt.close('all')





