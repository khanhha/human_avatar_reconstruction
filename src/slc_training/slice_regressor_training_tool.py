import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import shutil
import sys
from collections import defaultdict
from sklearn.cluster import KMeans
from slc_training.slice_regressor_dtree import RBFNet
from common import util
from slc_training.slice_train_util import load_slc_contours, load_slice_data, load_bad_slice_names

def slice_model_config():
    config = defaultdict(dict)

    config['Shoulder'] = {'n_cluster':12, 'n_output':12, 'no_regress_at_outputs':[]}

    config['Armscye'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[]}
    config['Bust'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[]}
    config['Aux_UnderBust_Bust_0'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[]}
    config['UnderBust'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[]}
    config['Crotch'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[0, 7]}

    config['Aux_Crotch_Hip_0'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[0, 7]}
    config['Aux_Crotch_Hip_1'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[0, 7]}
    config['Hip'] = {'n_cluster':12, 'n_output':10, 'no_regress_at_outputs':[0, 7]}

    config['UnderCrotch'] = {'n_cluster':12, 'n_output':9, 'no_regress_at_outputs':[]}
    config['Aux_Knee_UnderCrotch_3'] = {'n_cluster':12, 'n_output':9, 'no_regress_at_outputs':[]}
    config['Aux_Knee_UnderCrotch_2'] = {'n_cluster':12, 'n_output':9, 'no_regress_at_outputs':[]}
    config['Aux_Knee_UnderCrotch_1'] = {'n_cluster':12, 'n_output':9, 'no_regress_at_outputs':[]}
    config['Aux_Knee_UnderCrotch_0'] = {'n_cluster':12, 'n_output':9, 'no_regress_at_outputs':[]}
    config['Knee']                   = {'n_cluster':12, 'n_output':9, 'no_regress_at_outputs':[]}
    return config

def print_statistic(X, Y):
    mean = np.mean(Y, axis=0)
    median = np.median(Y, axis=0)
    std = np.std(Y, axis=0)
    max = np.max(Y, axis=0)
    min = np.min(Y, axis=0)
    np.set_printoptions(suppress=True)
    print('Target Y statistics: ')
    for i in range(mean.shape[0]):
        print(f'\tY[{i}] mean, median, std, max, min = {mean[i]}, {median[i]}, {std[i]}, {max[i]}, {min[i]}' )

    print('nan count: ', np.isnan(X).flatten().sum())
    print('nan count: ', np.isnan(Y).flatten().sum())
    print('inf count: ', np.isinf(X).flatten().sum())
    print('inf count: ', np.isinf(Y).flatten().sum())

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="root directory contains all slice directory")
    ap.add_argument("-c", "--code", required=True, help="root directory contains all slice code directory")
    ap.add_argument("-d", "--debug", required=True, help="debug directory to output inference result")
    ap.add_argument("-m", "--model", required=True, help="output directory to save model")
    ap.add_argument("-b", "--bad_slice_dir", required=True, help="folder that contains text files storing bad slice name for each slice type")
    ap.add_argument("-ids", "--slc_ids", required=True, help="which slice do you want to train? set it to 'all' or the following pattern: slice_0,slice_1,slice_2 ")
    ap.add_argument("-test_infer", "--test_inference", required=True, help="do inference on the test data using the trained model?")
    ap.add_argument("-train_infer", "--train_inference", required=True, help="do inference on the training data using the trained model?")

    args = vars(ap.parse_args())
    ALL_SLC_DIR  = args['input']
    CODE_DIR  = args['code']
    DEBUG_DIR  = args['debug']
    MODEL_DIR_ROOT  = args['model']
    BAD_SLICE_DIR = args['bad_slice_dir']
    slc_ids = args['slc_ids']
    do_test_infer = int(args['test_inference'])   > 0
    do_train_infer = int(args['train_inference']) > 0

    all_slc_ids = [path.stem for path in Path(ALL_SLC_DIR).glob('./*')]
    if slc_ids == 'all' or slc_ids == 'All':
        slc_ids = all_slc_ids
    else:
        slc_ids = slc_ids.split(',')
        for id in slc_ids:
            assert id in all_slc_ids, f'{id}: unrecognized slice id'

    model_configs = slice_model_config()

    code_type = 'fourier'

    MODEL_DIR = f'{MODEL_DIR_ROOT}/{code_type}/'
    os.makedirs(MODEL_DIR, exist_ok=True)

    #load data from disk
    for SLC_DIR in Path(ALL_SLC_DIR).glob('*'):
        slc_id = SLC_DIR.stem
        if slc_id in slc_ids:

            print(f'\nslice: {slc_id}')

            model_config = model_configs[slc_id]
            K = model_config['n_cluster'] if 'n_cluster' in model_config else 12

            slc_id = SLC_DIR.stem
            bad_slc_names = load_bad_slice_names(BAD_SLICE_DIR, slc_id)

            print('\tstart loading data')
            SLC_CODE_DIR = f'{CODE_DIR}/{slc_id}/'
            X, Y, W, D, slc_idx_names = load_slice_data(SLC_CODE_DIR, bad_slc_names)
            X = np.reshape(X, (-1, 1))

            contours = load_slc_contours(SLC_DIR)

            if code_type == 'fourier':
                n_output = None
                no_regress_at_outputs = []
            else:
                n_output = model_config['n_output'] if 'n_output' in model_config else 10
                no_regress_at_outputs = model_config['no_regress_at_outputs'] if 'no_regress_at_outputs' in model_config else [0, 7]

            print('\tstart training')
            net = RBFNet(slc_id=slc_id, n_cluster=K, n_output=n_output, no_regress_at_outputs=no_regress_at_outputs, debug_mode=True)
            net.fit(X, Y)

            MODEL_PATH = f'{MODEL_DIR}/{SLC_DIR.stem}.pkl'
            net.save_to_path(MODEL_PATH)
            print(f'\tsaved model to file {MODEL_PATH}')
            net_1 = RBFNet.load_from_path(MODEL_PATH)

            if do_test_infer:
                OUTPUT_DEBUG_DIR_TEST = f'{DEBUG_DIR}/{SLC_DIR.stem}_prediction/test/'
                shutil.rmtree(OUTPUT_DEBUG_DIR_TEST, ignore_errors=True)
                os.makedirs(OUTPUT_DEBUG_DIR_TEST, exist_ok=True)

                for i in range(len(net.test_idxs)):
                    idx = net_1.test_idxs[i]

                    print('processing test idx: ', idx)
                    pred = net_1.predict(np.expand_dims(X[idx, :], axis=0))[0, :]

                    w = W[idx]
                    d = D[idx]
                    contour = contours[slc_idx_names[idx]]
                    center = util.contour_center(contour[0, :], contour[1, :])

                    if code_type == 'fourier':
                        res_contour = util.reconstruct_contour_fourier(pred.flatten())
                        res_contour_org = np.copy(res_contour)
                        res_range_x = np.max(res_contour[0,:]) - np.min(res_contour[0,:])
                        range_x = np.max(contour[0,:]) - np.min(contour[0,:])
                        scale_x = range_x / res_range_x

                        res_range_y = np.max(res_contour[1,:]) - np.min(res_contour[1,:])
                        range_y = np.max(contour[1,:]) - np.min(contour[1,:])
                        scale_y = range_y / res_range_y

                        res_contour[0,:] *= scale_x
                        res_contour[1,:] *= scale_y

                        res_contour_org *= max(scale_x, scale_y)

                        error = np.sqrt(np.mean(np.square(res_contour_org - res_contour)))
                        print(f'\tmean error uniform scale vs non-uniform scale = {error}')
                        #print(f'prediction = {pred}')
                    else:
                        if util.is_leg_contour(slc_id):
                            res_contour = util.reconstruct_leg_slice_contour(pred, d, w)
                        else:
                            res_contour = util.reconstruct_torso_slice_contour(pred, d, w, mirror=True)

                    res_contour[0, :] += center[0]
                    res_contour[1, :] += center[1]
                    last_p = res_contour[:,0].reshape(2,1)
                    res_contour = np.concatenate([res_contour, last_p], axis=1)

                    res_contour_org[0, :] += center[0]
                    res_contour_org[1, :] += center[1]
                    last_p = res_contour_org[:,0].reshape(2,1)
                    res_contour_org = np.concatenate([res_contour_org, last_p], axis=1)

                    plt.clf()
                    plt.axes().set_aspect(1)
                    plt.plot(contour[0, :], contour[1, :], '-b')

                    plt.plot(res_contour_org[0, :], res_contour_org[1, :], '-y')

                    plt.plot(res_contour[0, :], res_contour[1, :], '-r')
                    plt.plot(res_contour[0, :], res_contour[1, :], '+r')
                    plt.plot(res_contour[0, 0], res_contour[1, 0], '+r', ms=20)
                    plt.plot(res_contour[0, 2], res_contour[1, 2], '+g', ms=20)
                    plt.savefig(f'{OUTPUT_DEBUG_DIR_TEST}{idx}.png')
                    #plt.show()

            # if do_train_infer:
            #     OUTPUT_DEBUG_DIR_TRAIN = f'{DEBUG_DIR}/{SLC_DIR.stem}_prediction/train/'
            #     shutil.rmtree(OUTPUT_DEBUG_DIR_TRAIN, ignore_errors=True)
            #     os.makedirs(OUTPUT_DEBUG_DIR_TRAIN, exist_ok=True)
            #
            #     for i in range(len(net.train_idxs)):
            #         idx = net.train_idxs[i]
            #         print('processing train idx: ', idx)
            #         w = W[idx]
            #         d = D[idx]
            #         contour = contours[idx]
            #         pred = net_1.predict(np.expand_dims(X[idx, :], axis=0))[0, :]
            #         if code_type == 'fourier':
            #             res_contour = util.reconstruct_contour_fourier(pred.flatten())
            #             res_range_x = np.max(res_contour[0, :]) - np.min(res_contour[0, :])
            #             range_x = np.max(contour[0, :]) - np.min(contour[0, :])
            #             scale_x = range_x / res_range_x
            #
            #             res_range_y = np.max(res_contour[1, :]) - np.min(res_contour[1, :])
            #             range_y = np.max(contour[1, :]) - np.min(contour[1, :])
            #             scale_y = range_y / res_range_y
            #
            #             res_contour[0, :] *= scale_x
            #             res_contour[1, :] *= scale_y
            #         else:
            #             res_contour = util.reconstruct_torso_slice_contour(pred, d, w, mirror=True)
            #
            #         center = util.contour_center(contour[0, :], contour[1, :])
            #         res_contour[0, :] += center[0]
            #         res_contour[1, :] += center[1]
            #         last_p = res_contour[:,0].reshape(2,1)
            #         res_contour = np.concatenate([res_contour, last_p], axis=1)
            #         plt.clf()
            #         plt.axes().set_aspect(1)
            #         #plt.plot(contour[0, :], contour[1, :], '-b')
            #         plt.plot(res_contour[0, :], res_contour[1, :], '-r')
            #         plt.plot(res_contour[0, :], res_contour[1, :], '+r')
            #         plt.savefig(f'{OUTPUT_DEBUG_DIR_TRAIN}{idx}.png')
