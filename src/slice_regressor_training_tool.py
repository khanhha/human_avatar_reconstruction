import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import shutil
import sys
from collections import defaultdict
from sklearn.cluster import KMeans
from slice_regressor_dtree import RBFNet
from common import util

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

def load_bad_slice_names(DIR, slc_id):
    txt_path = None
    for path in Path(DIR).glob('*.*'):
        if slc_id == path.stem:
            txt_path = path
            break

    if txt_path is None:
        print(f'missing bad slice path of slice {slc_id}', file=sys.stderr)
        return ()
    else:
        names = set()
        with open(str(txt_path), 'r') as file:
            for name in file.readlines():
                name = name.replace('\n','')
                names.add(name)
        return names

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

def load_slice_data(SLC_CODE_DIR, bad_slc_names):
    slc_names = []
    all_paths = [path for path in Path(SLC_CODE_DIR).glob('*.*')]
    X, Y = [], []
    W, D = [], []
    for path in all_paths:
        if path.stem in bad_slc_names:
            continue
        with open(str(path), 'rb') as file:
            record = pickle.load(file)
            w = record['W']
            d = record['D']

            if w == 0.0 or d == 0.0:
                print('zero w or d: ', w, d, file=sys.stderr)
                continue

            feature = record['Code']

            if np.isnan(feature).flatten().sum() > 0:
                print(f'nan feature: {path}', file=sys.stderr)
                continue

            if np.isnan(X).flatten().sum() > 0:
                print(f'nan X: {path}', file=sys.stderr)
                continue

            if np.isinf(feature).flatten().sum() > 0:
                print(f'inf feature: {path}', file=sys.stderr)
                continue

            if np.isinf(X).flatten().sum() > 0:
                print(f'inf X: {path}', file=sys.stderr)
                continue

            slc_names.append(path.stem)

            #W and D arrays are just of the sake of test inference
            W.append(w)
            D.append(d)

            X.append(w / d)
            Y.append(feature)

    print_statistic(X, Y)

    return np.array(X), np.array(Y), W, D, slc_names

def load_slc_contours(SLC_DIR):
    contours = {}
    for path in Path(SLC_DIR).glob('*.pkl'):
        with open(str(path), 'rb') as file:
            record = pickle.load(file)
            contours[path.stem] = record['cnt']
    return contours

def normalize_contour(X,Y, center):
    X = X-center[0]
    Y = Y-center[1]
    dsts = np.sqrt(np.square(X) + np.square(Y))
    mean_dst = np.max(dsts)
    X = X / mean_dst
    Y = Y / mean_dst
    return X, Y

import os
def plot_contour_correrlation(IN_DIR, DEBUG_DIR, K):
    ratios = []
    contours = []
    for path in Path(DEBUG_DIR).glob('*.*'):
        os.remove(str(path))

    for path in Path(IN_DIR).glob("*.*"):
        with open(str(path), 'rb') as file:
            data = pickle.load(file)
            w = data['W']
            d = data['D']
            ratios.append(w/d)
            contours.append(data['cnt'])

    n_contour = len(ratios)
    ratios = np.array(ratios).reshape(n_contour, 1)
    kmeans = KMeans(n_clusters=K)
    cnt_labels =  kmeans.fit_predict(ratios)
    for l in kmeans.labels_:
        l_contour_idxs = np.argwhere(cnt_labels==l)[:,0]
        cls_center = kmeans.cluster_centers_[l]
        cluster_mask = (cnt_labels == l)
        points = ratios[cluster_mask].flatten()
        std = np.std(points)
        plt.clf()
        plt.axes().set_aspect(1.0)
        #cnt_centers = [util.contour_center(contours[i][:,0], contours[i][:,1]) for i in l_contour_idxs]
        #cnt_centers = np.array(cnt_centers)
        #mean_center = np.mean(cnt_centers, axis=0)
        for idx in l_contour_idxs:
            contour = contours[idx]
            X = contour[0,:]
            Y = contour[1,:]
            X,Y = normalize_contour(X, Y, util.contour_center(X, Y))
            X0 = X[0].reshape(-1)
            X = np.concatenate([X, X0], axis=0)
            Y0 = Y[0].reshape(-1)
            Y = np.concatenate([Y, Y0], axis=0)
            plt.plot(X, Y, '-b')
            plt.title(f'n_contour_in_cluster = {len(l_contour_idxs)}\n center = {cls_center}, inertia={std}')

        plt.savefig(f'{DEBUG_DIR}/label_{l}.png')


def plot_all_contour_correlation():
    #plot correlation
    # print('plotting correlation k-means')
    # for path in Path(IN_DIR).glob('*'):
    #     slc_id = path.stem
    #     if slc_id in slc_ids:
    #
    #         model_config = model_configs[slc_id]
    #         K = model_config['n_cluster'] if 'n_cluster' in model_config else 12
    #
    #         CORR_DIR = f'{DEBUG_DIR}{slc_id}_correlation/'
    #         os.makedirs(CORR_DIR, exist_ok=True)
    #         plot_contour_correrlation(str(path), CORR_DIR, K)
    # exit()
    pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    ap.add_argument("-c", "--code", required=True, help="contour code")
    ap.add_argument("-d", "--debug", required=True, help="input meta data file")
    ap.add_argument("-m", "--model", required=True, help="input meta data file")
    ap.add_argument("-b", "--bad_slice_dir", required=True, help="input meta data file")
    ap.add_argument("-ids", "--slc_ids", required=True, help="input meta data file")
    ap.add_argument("-test_infer", "--test_inference", required=True, help="input meta data file")
    ap.add_argument("-train_infer", "--train_inference", required=True, help="input meta data file")

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
    if slc_ids == 'all':
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
            model_config = model_configs[slc_id]
            K = model_config['n_cluster'] if 'n_cluster' in model_config else 12

            slc_id = SLC_DIR.stem
            bad_slc_names = load_bad_slice_names(BAD_SLICE_DIR, slc_id)

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

            print('start training slice mode: ', slc_id)
            net = RBFNet(slc_id=slc_id, n_cluster=K, n_output=n_output, no_regress_at_outputs=no_regress_at_outputs, debug_mode=True)
            net.fit(X, Y)

            MODEL_PATH = f'{MODEL_DIR}/{SLC_DIR.stem}.pkl'
            net.save_to_path(MODEL_PATH)
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