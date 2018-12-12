import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import argparse
import shutil
from collections import defaultdict
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adadelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from xgboost import XGBRegressor

from dtreeviz.trees import *

import src.util as util
from  src.error_files import mpii_error_slices, ucsc_error_slices


def kmeans(X, k):
    """Performs k-means clustering for 1D input
    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters
    Returns:
        ndarray -- A kx1 array of final cluster centers
    """

    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False

    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)

        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds

class RBFNeuralNetwork():
    def __init__(self, n_cluster = 0, n_output = 0, no_regress_at_outputs= []):
        self.n_cluster = n_cluster
        self.n_output = n_output
        self.no_regress_at_outputs = no_regress_at_outputs
        for k in self.no_regress_at_outputs:
            assert 0<=k and k < n_output

    @staticmethod
    def load_from_path(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
            return obj

    def save_to_path(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def fit(self, X, Y):
        self.cluster_data(X)
        X_1 = self.transform_data(X)

        nan_mask = np.isnan(X_1)
        print(f'transformed_data: nan count: {np.sum(nan_mask[:])}')

        N = X.shape[0]
        X_train, X_test,  train_idxs, test_idxs = train_test_split(X_1, np.arange(N), test_size=0.2, shuffle=True)
        X_train, X_valid, train_idxs, valid_idxs = train_test_split(X_train, train_idxs, test_size=0.2, shuffle=True)

        self.train_idxs = train_idxs
        self.valid_idxs = valid_idxs
        self.test_idxs = test_idxs
        Y_train = Y[train_idxs]
        Y_valid = Y[valid_idxs]
        Y_test = Y[test_idxs]

        input = Input(shape=(self.n_cluster,))
        x = Dense(units=32, activation='relu')(input)
        x = Dense(units=64, activation='relu')(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dense(units=self.n_output)(x)
        model = Model(input=input, output = x)
        model.summary()
        model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mean_squared_error'])
        epochs = 300
        history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=epochs, batch_size=32)
        self.regressor = model
        plt.plot(np.arange(epochs), history.history['loss'], '-b')
        plt.plot(np.arange(epochs), history.history['val_loss'], '-r')
        plt.show()

        # calc median of the first and last curvature for each cluster
        self.output_cluster_median = np.zeros(shape=(self.n_cluster, self.n_output))
        for cluster in range(self.n_cluster):
            for output_idx in range(self.n_output):
                features = Y[self.training_clusters == cluster]
                self.output_cluster_median[cluster, output_idx] = np.median(features[:, output_idx])

    def loss(self, X, Y):
        Y_hat = self.regressor.predict(X)
        for i in range(10):
            print('')
            print('y_hat', Y_hat[i,:])
            print('y    ', Y[i,:])
        l = np.sqrt(np.mean((Y-Y_hat)**2))
        return l

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        cluster_ids = self.kmeans_cls.predict(X)

        X_1 = self.transform_data(X)
        preds = self.regressor.predict(X_1)

        for i in range(preds.shape[0]):
            preds[i, self.no_regress_at_outputs] = self.output_cluster_median[cluster_ids[i], self.no_regress_at_outputs]

        return preds

    def rbf(self, x, c, s):
        #print(2 * s ** 2)
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    def transform_data(self, X):
        X_1 = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s)[0] for c, s, in zip(self.cluster_mean, self.cluster_std)])
            X_1.append(a)
        X_1 = np.array(X_1)
        return X_1

    def cluster_data(self, X):
        self.kmeans_cls = KMeans(n_clusters=self.n_cluster, max_iter=1000, tol=1e-6, random_state=100).fit(X)
        self.training_clusters = self.kmeans_cls.predict(X)

        # calculate standard variance
        self.cluster_std = np.zeros(len(self.kmeans_cls.labels_), dtype=np.float32)
        self.cluster_mean = self.kmeans_cls.cluster_centers_.flatten()
        for l in range(self.n_cluster):
            cluster_mask = (self.training_clusters== l)
            points = X[cluster_mask].flatten()
            self.cluster_std[l] = np.std(points)

class RBFNet():
    def __init__(self, slc_id, n_cluster = 0, n_output = 0, no_regress_at_outputs= [], debug_mode = False):
        self.slc_id = slc_id
        self.debug_mode = debug_mode
        self.n_cluster = n_cluster
        self.no_regress_at_outputs = no_regress_at_outputs

    @staticmethod
    def load_from_path(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
            return obj
            # self.n_cluster = data.n_cluster
            # self.n_output = data.n_output
            # self.no_regress_at_outputs = data.no_regress_at_outputs
            #
            # self.kmeans_cls = data.kmeans_cls
            # self.training_clusters = data.training_clusters
            #
            # self.cluster_std = data.cluster_std
            # self.cluster_mean = data.cluster_mean

    def save_to_path(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def fit(self, X, Y):
        self.cluster_data(X)
        X_1 = self.transform_data(X)

        nan_mask = np.isnan(X_1)
        nan_mask = np.sum(nan_mask, axis=1) > 0
        print(f'transformed_data: nan count: {np.sum(nan_mask[:])}')
        if np.sum(nan_mask[:]) > 0:
            X_1 = X_1[~nan_mask, :]
            Y   = Y[~nan_mask, :]

        N = X_1.shape[0]
        X_train, X_test, train_idxs, test_idxs = train_test_split(X_1, np.arange(N), test_size=0.2, shuffle=True, random_state=200)
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        Y_train = Y[train_idxs]
        Y_test = Y[test_idxs]

        if self.debug_mode == True:
            self.X_train = X_train
            self.Y_train = Y_train

        search = False
        if search:
            parameters = {'n_estimators':[40, 60, 80, 100, 120], 'min_samples_leaf':[20, 40, 60, 80, 100, 120]}
            regressor = ExtraTreesRegressor(random_state=200)
            clf = GridSearchCV(regressor, parameters, cv = 5)
            clf.fit(X_train, Y_train)
            print(clf.best_estimator_)
            self.regressor = clf.best_estimator_
        else:
            #self.regressor = ExtraTreesRegressor(random_state=200, n_estimators=100, min_samples_leaf=100)
            self.regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, min_samples_leaf=50))
            self.regressor.fit(X_train, Y_train)

        #self.regressor = XGBRegressor().fit(X_train, Y_train)
        #self.regressor =  GradientBoostingRegressor(**params).fit(X_train, Y_train)
        print('regression score on train set:  ', self.regressor.score(X_train, Y_train))
        print('regression score on test set: ', self.regressor.score(X_test, Y_test))
        print('regression score on test mse loss: ', self.loss(X_test, Y_test))

        # calc median of the first and last curvature for each cluster
        n_ouput = Y.shape[1]
        self.output_cluster_median = np.zeros(shape=(self.n_cluster, n_ouput))
        for cluster in range(self.n_cluster):
            for output_idx in range(n_ouput):
                features = Y[self.training_clusters == cluster]
                self.output_cluster_median[cluster, output_idx] = np.median(features[:, output_idx])

    def loss(self, X, Y):
        Y_hat = self.regressor.predict(X)
        l = np.sqrt(np.mean((Y-Y_hat)**2))
        return l

    def visualize_regressor(self, DIR):
        is_viz = True
        if is_viz:
            for i, forest in enumerate(self.regressor.estimators_):
                for j, tree in enumerate(forest.estimators_):
                    viz = dtreeviz(tree,
                                   self.X_train,
                                   self.Y_train[:,i],
                                   feature_names=[str(i) for i in range(self.n_cluster)],
                                   target_name=f'target_{i}')
                    viz.save(f'{DIR}/{slc_id}/{i}_{j}.svg')

    def visualize_an_obsevation(self, x, DIR, file_name = ''):
        X = np.expand_dims(x, axis=0)
        X_1 = self.transform_data(X)
        for i, forest in enumerate(self.regressor.estimators_):
            for j, tree in enumerate(forest.estimators_):
                viz = dtreeviz(tree,
                               self.X_train,
                               self.Y_train[:, i],
                               feature_names=[str(i) for i in range(self.n_cluster)],
                               target_name=f'target_{i}',
                               X = X_1.flatten())
                viz.save(f'{DIR}/{self.slc_id}/{file_name}_{i}_{j}.svg')

        return self.regressor.predict(X_1)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        cluster_ids = self.kmeans_cls.predict(X)

        X_1 = self.transform_data(X)
        preds = self.regressor.predict(X_1)

        for i in range(preds.shape[0]):
            preds[i, self.no_regress_at_outputs] = self.output_cluster_median[cluster_ids[i], self.no_regress_at_outputs]

        return preds

    def rbf(self, x, c, s):
        #print(2 * s ** 2)
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    def transform_data(self, X):
        X_1 = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s)[0] for c, s, in zip(self.cluster_mean, self.cluster_std)])
            X_1.append(a)
        X_1 = np.array(X_1)
        return X_1

    def cluster_data(self, X):
        rd = 100
        while True:
            self.kmeans_cls = KMeans(n_clusters=self.n_cluster, max_iter=1000, tol=1e-6, random_state=rd).fit(X)
            self.training_clusters = self.kmeans_cls.predict(X)

            # calculate standard variance
            self.cluster_std = np.zeros(self.n_cluster, dtype=np.float32)
            self.cluster_mean = self.kmeans_cls.cluster_centers_.flatten()
            for l in range(self.n_cluster):
                cluster_mask = (self.training_clusters== l)
                points = X[cluster_mask].flatten()
                self.cluster_std[l] = np.std(points)

            zero_mask = np.isclose(self.cluster_std, 0.0)
            if np.sum(zero_mask) == 0:
                break
            else:
                print(f'cluster std is zero, reduced cluster: {self.n_cluster - 1}. repeat again', file=sys.stderr)
                self.n_cluster -= 1
                rd = np.random.randint(0, 10000)

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
            X,Y = normalize_contour(X,Y, util.contour_center(X,Y))
            X0 = X[0].reshape(-1)
            X = np.concatenate([X, X0], axis=0)
            Y0 = Y[0].reshape(-1)
            Y = np.concatenate([Y, Y0], axis=0)
            plt.plot(X, Y, '-b')
            plt.title(f'n_contour_in_cluster = {len(l_contour_idxs)}\n center = {cls_center}, inertia={std}')

        plt.savefig(f'{DEBUG_DIR}/label_{l}.png')

def slice_model_config():
    config = defaultdict(set)

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

import sys
from copy import copy, deepcopy
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
    IN_DIR  = args['input']
    CODE_DIR  = args['code']
    DEBUG_DIR  = args['debug']
    MODEL_DIR_ROOT  = args['model']
    BAD_SLICE_DIR = args['bad_slice_dir']
    slc_ids = args['slc_ids']
    do_test_infer = int(args['test_inference'])   > 0
    do_train_infer = int(args['train_inference']) > 0

    all_slc_ids = [path.stem for path in Path(IN_DIR).glob('./*')]
    if slc_ids == 'all':
        slc_ids = all_slc_ids
    else:
        slc_ids = slc_ids.split(',')
        for id in slc_ids:
            assert id in all_slc_ids, f'{id}: unrecognized slice id'

    model_configs = slice_model_config()

    code_type = Path(CODE_DIR).stem
    MODEL_DIR = f'{MODEL_DIR_ROOT}/{code_type}/'
    os.makedirs(MODEL_DIR, exist_ok=True)

    #load data from disk
    for SLC_DIR in Path(IN_DIR).glob('*'):

        slc_id = SLC_DIR.stem

        if slc_id in slc_ids:

            MODEL_PATH = f'{MODEL_DIR}/{SLC_DIR.stem}.pkl'

            OUTPUT_DEBUG_DIR_TRAIN = f'{DEBUG_DIR}/{SLC_DIR.stem}_prediction/train/'
            OUTPUT_DEBUG_DIR_TEST = f'{DEBUG_DIR}/{SLC_DIR.stem}_prediction/test/'
            shutil.rmtree(OUTPUT_DEBUG_DIR_TEST, ignore_errors=True)
            shutil.rmtree(OUTPUT_DEBUG_DIR_TRAIN, ignore_errors=True)
            os.makedirs(OUTPUT_DEBUG_DIR_TRAIN, exist_ok=True)
            os.makedirs(OUTPUT_DEBUG_DIR_TEST, exist_ok=True)

            model_config = model_configs[slc_id]
            K = model_config['n_cluster'] if 'n_cluster' in model_config else 12

            #collect all slices
            X = []; Y = []; W = [];   D = [];  contours = []
            slc_id = SLC_DIR.stem
            print('start training slice mode: ', slc_id)
            bad_slc_names = load_bad_slice_names(BAD_SLICE_DIR, slc_id)

            SLC_CODE_DIR = f'{CODE_DIR}/{slc_id}/'

            all_paths = [path for path in Path(SLC_CODE_DIR).glob('*.*')]
            for path in all_paths:

                if path.stem in bad_slc_names:
                    continue

                with open(str(path),'rb') as file:
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

                    X.append(w/d)
                    Y.append(feature)
                    W.append(w)
                    D.append(d)

                slc_path = f'{SLC_DIR}/{path.stem}.pkl'
                with open(slc_path, 'rb') as file:
                    record = pickle.load(file)
                    contours.append(record['cnt'])

            N = len(X)
            X = np.array(X)
            Y = np.array(Y)
            X = np.reshape(X, (-1, 1))

            #print_statistic(X, Y)

            if code_type == 'fourier':
                n_output = None
                no_regress_at_outputs = []
            else:
                n_output = model_config['n_output'] if 'n_output' in model_config else 10
                no_regress_at_outputs = model_config['no_regress_at_outputs'] if 'no_regress_at_outputs' in model_config else [0, 7]

            use_tree = True
            if use_tree:
                net = RBFNet(slc_id=slc_id, n_cluster=K, n_output=n_output, no_regress_at_outputs=no_regress_at_outputs, debug_mode=True)
                net.fit(X, Y)
                net.save_to_path(MODEL_PATH)
                #debug
                #VIZ_DEBUG_DIR = f'{DEBUG_DIR}/tree_viz/'
                #test_id = 45
                #preds = net.visualize_an_obsevation(X[test_id,:], VIZ_DEBUG_DIR, test_id.__str__())
                #print(f'prediction result of {test_id}: {preds}')
            else:
                net = RBFNeuralNetwork(n_cluster=K, n_output=n_output, no_regress_at_outputs=no_regress_at_outputs)
                net.fit(X, Y)
                net.save_to_path(MODEL_PATH)

            net_1 = RBFNet.load_from_path(MODEL_PATH)

            if do_test_infer:

                for i in range(len(net.test_idxs)):
                    idx = net_1.test_idxs[i]

                    print('processing test idx: ', idx)
                    pred = net_1.predict(np.expand_dims(X[idx, :], axis=0))[0, :]

                    w = W[idx]
                    d = D[idx]
                    contour = contours[idx]
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
                        print(f'prediction = {pred}')

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

            if do_train_infer:
                for i in range(len(net.train_idxs)):
                    idx = net.train_idxs[i]
                    print('processing train idx: ', idx)
                    w = W[idx]
                    d = D[idx]
                    contour = contours[idx]
                    pred = net_1.predict(np.expand_dims(X[idx, :], axis=0))[0, :]
                    if code_type == 'fourier':
                        res_contour = util.reconstruct_contour_fourier(pred.flatten())
                        res_range_x = np.max(res_contour[0, :]) - np.min(res_contour[0, :])
                        range_x = np.max(contour[0, :]) - np.min(contour[0, :])
                        scale_x = range_x / res_range_x

                        res_range_y = np.max(res_contour[1, :]) - np.min(res_contour[1, :])
                        range_y = np.max(contour[1, :]) - np.min(contour[1, :])
                        scale_y = range_y / res_range_y

                        res_contour[0, :] *= scale_x
                        res_contour[1, :] *= scale_y
                    else:
                        res_contour = util.reconstruct_torso_slice_contour(pred, d, w, mirror=True)

                    center = util.contour_center(contour[0, :], contour[1, :])
                    res_contour[0, :] += center[0]
                    res_contour[1, :] += center[1]
                    last_p = res_contour[:,0].reshape(2,1)
                    res_contour = np.concatenate([res_contour, last_p], axis=1)
                    plt.clf()
                    plt.axes().set_aspect(1)
                    #plt.plot(contour[0, :], contour[1, :], '-b')
                    plt.plot(res_contour[0, :], res_contour[1, :], '-r')
                    plt.plot(res_contour[0, :], res_contour[1, :], '+r')
                    plt.savefig(f'{OUTPUT_DEBUG_DIR_TRAIN}{idx}.png')

