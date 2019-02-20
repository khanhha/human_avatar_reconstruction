import argparse
import pickle
import os
from slc_training.slice_train_util import load_bad_slice_names, load_slice_data_1, SlcData
import common.util as util
import matplotlib.pyplot as plt
import numpy as np
import itertools

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

def load_model_datas(dir, model_ids, slc_ids):
    models = []
    for model_id in model_ids:
        slc_models = []

        for slc_id in slc_ids:
            model_path = os.path.join(*[dir, model_id, f'{slc_id}.pkl'])
            with open(model_path, 'rb') as file:
                model = pickle.load(file)

            assert is_valid_model(model['model'], slc_id) == 1

            slc_models.append(model)

        models.append(slc_models)

    return models

def load_all_slice_data(models, all_slc_dir, all_feature_dir, bad_slc_dir):

    load_slc_ids = set()
    for model in models:
        load_slc_ids.add(model.slc_id)
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

def load_test_names_from_all_models(models_data):
    test_names = set()
    for model_data in models_data:
        if len(test_names) == 0:
            test_names = set(model_data['test_fnames'])
        else:
            test_names = test_names.intersection(set(model_data['test_fnames']))

    return list(test_names)

def all_used_ids(models):
    load_slc_ids = set()

    for model in models:
        load_slc_ids.add(model.slc_id)
        for id in model.slc_model_input_ids:
            load_slc_ids.add(id)

    return list(load_slc_ids)

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
    ap.add_argument("-model_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-model_type_ids", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-slc_ids", required=True, type=str, help="root directory contains all slice code directory")
    args = ap.parse_args()

    #debug_dir = os.path.join(args.debug_dir, 'Crotch_to_Hip')
    os.makedirs(args.debug_dir, exist_ok=True)

    model_type_ids = args.model_type_ids.split(',')
    slc_ids = args.slc_ids.split(',')

    models_data = load_model_datas(args.model_dir, model_type_ids, slc_ids)

    models = []
    for slc_model_datas in models_data:
        slc_models = [model_data['model'] for model_data in slc_model_datas]
        models.append(slc_models)

    load_slc_ids = all_used_ids(list(itertools.chain.from_iterable(models)))

    all_slc_data = load_all_slice_data(list(itertools.chain.from_iterable(models)), args.slc_dir, args.feature_dir, args.bad_slc_dir)

    #collect file names shared by all slice data
    print('collect file names shared by all slice')
    #shared_fnames = load_shared_file_names(models_0, models_1, all_slc_data)
    shared_fnames = load_test_names_from_all_models(list(itertools.chain.from_iterable(models_data)))
    print(f'n test names = {len(shared_fnames)}')

    n_models = len(model_type_ids)
    n_slc = len(slc_ids)

    #predict result by all models
    print('calculate prediction by all models')
    models_preds = []
    for i in range(n_models):
        preds = []
        for j in range(n_slc):
            model = models[i][j]
            slc_id = slc_ids[j]
            in_slc_data = [all_slc_data[id] for id in model.slc_model_input_ids]
            out_slc_data = all_slc_data[slc_id]

            X, _ = SlcData.build_training_data(in_slc_data, out_slc_data, shared_fnames)

            P = model.predict(X)
            preds.append(P)

        models_preds.append(preds)

    fontsize = 2
    for file_idx, name in enumerate(shared_fnames):

        model_contours = []
        res_model_contours = []

        for i in range(n_models):
            res_slc_contours = []
            slc_contours = []

            slc_preds = models_preds[i]

            for j in range(n_slc):

                slc_id = slc_ids[j]
                SLC_DIR = os.path.join(args.slc_dir, slc_id)

                slc_contours.append(normalize_contour(load_contour(SLC_DIR, name), resolution = 20))

                pred = slc_preds[j][file_idx, :]
                res_slc_contours.append(util.reconstruct_contour_fourier(pred.flatten()))

            res_model_contours.append(res_slc_contours)
            model_contours.append(slc_contours)

        fig, axs = plt.subplots(n_models, n_slc)
        axs = axs.reshape(n_models, n_slc)

        for i in range(n_models):

            ax = axs[i, 0]
            ax.set_ylabel(f'{model_type_ids[i]}', rotation=1, size=fontsize+2, color='r', x=0)

            for j  in range(n_slc):
                contour = model_contours[i][j]
                res_contour = res_model_contours[i][j]

                contour = np.concatenate([contour[:,:], contour[:,0].reshape(2,1)], axis=1)
                res_contour = np.concatenate([res_contour[:, :], res_contour[:, 0].reshape(2, 1)], axis=1)

                inputs = models[i][j].slc_model_input_ids
                slc_id = slc_ids[j]
                ax = axs[i,j]
                ax.set_aspect(1.0)
                ax.plot(contour[0,:], contour[1,:], '-b', linewidth=0.8)
                ax.plot(res_contour[0, :], res_contour[1, :], '-r', linewidth=0.8)
                ax.set_title(f'model name = {slc_id} \ninputs = {align_ids(inputs)}', fontsize=fontsize, loc='left', y=0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

        title = f'{name} \n green: ground truth. red: prediction'
        #fig.suptitle(title, fontsize=fontsize+2)
        #fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        #plt.show()
        plt.savefig(os.path.join(args.debug_dir, f'{name}.svg'), format='svg', dpi=2000)
        plt.close('all')





