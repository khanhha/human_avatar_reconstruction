import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import argparse
import shutil
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adadelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
import src.util as util

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


class RBFNet():
    def __init__(self, n_cluster = 0, n_output = 0, no_regress_at_ouputs= []):
        self.n_cluster = n_cluster
        self.n_output = n_output
        self.no_regress_at_outputs = no_regress_at_ouputs
        for k in self.no_regress_at_outputs:
            assert 0<=k and k < n_output

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

        N = X.shape[0]
        X_train, X_test, train_idxs, test_idxs = train_test_split(X_1, np.arange(N), test_size=0.2, shuffle=True)
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        Y_train = Y[train_idxs]
        Y_test = Y[test_idxs]

        self.regressor = ExtraTreesRegressor().fit(X_train, Y_train)
        print('regression score on train set:  ', self.regressor.score(X_train, Y_train))
        print('regression score on test set: ', self.regressor.score(X_test, Y_test))

        # calc median of the first and last curvature for each cluster
        self.output_cluster_median = np.zeros(shape=(self.n_cluster, self.n_output))
        for cluster in range(self.n_cluster):
            for output_idx in range(self.n_output):
                features = Y[self.training_clusters == cluster]
                self.output_cluster_median[cluster, output_idx] = np.median(features[:, output_idx])

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
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    def transform_data(self, X):
        X_1 = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s)[0] for c, s, in zip(self.cluster_mean, self.cluster_std)])
            X_1.append(a)
        X_1 = np.array(X_1)
        return X_1

    def cluster_data(self, X):
        self.kmeans_cls = KMeans(n_clusters=self.n_cluster, max_iter=1000, tol=1e-6).fit(X)
        self.training_clusters = self.kmeans_cls.predict(X)

        # calculate standard variance
        self.cluster_std = np.zeros(len(self.kmeans_cls.labels_), dtype=np.float32)
        self.cluster_mean = self.kmeans_cls.cluster_centers_.flatten()
        for l in range(self.n_cluster):
            cluster_mask = (self.training_clusters== l)
            points = X[cluster_mask].flatten()
            self.cluster_std[l] = np.std(points)

def normalize_contour(X,Y):
    #cx = np.mean(X)
    #cy = np.mean(Y)
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    idx_xmax, idx_xmin = np.argmax(X), np.argmin(X)
    cy = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    cx = X[idx_ymin]

    X = X-cx
    Y = Y-cy
    dsts = np.sqrt(np.square(X) + np.square(Y))
    mean_dst = np.max(dsts)
    X = X / mean_dst
    Y = Y / mean_dst
    return X, Y

import os
def plot_contour_correrlation(IN_DIR, DEBUG_DIR):
    ratios = []
    contours = []
    for path in Path(DEBUG_DIR).glob('*.*'):
        os.remove(path)

    for path in Path(IN_DIR).glob("*.*"):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            w = data['W']
            d = data['D']
            ratios.append(w/d)
            contours.append(data['cnt'])

    n_contour = len(ratios)
    K = 32
    kmeans = KMeans(n_clusters=K)
    cnt_labels =  kmeans.fit_predict(np.array(ratios).reshape(n_contour, 1))
    for l in kmeans.labels_:
        l_contour_idxs = np.argwhere(cnt_labels==l)[:,0]
        plt.clf()
        plt.axes().set_aspect(1.0)
        for idx in l_contour_idxs:
            contour = contours[idx]
            X = contour[0,:]
            Y = contour[1,:]
            X,Y = normalize_contour(X,Y)
            plt.plot(X, Y, '-b')
        #plt.show()
        plt.savefig(f'{DEBUG_DIR}/label_{l}.png')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input meta data file")
    args = vars(ap.parse_args())
    IN_DIR  = args['input']

    #print('plotting correlation')
    #CONTOUR_DEBUG_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/hip_correlation/'
    #plot_contour_correrlation(IN_DIR, CONTOUR_DEBUG_DIR)

    #load data from disk
    X = []
    Y = []
    W = []
    D = []
    contours = []
    for path in Path(IN_DIR).glob('*.*'):
        with open(path,'rb') as file:
            record = pickle.load(file)
            w = record['W']
            d = record['D']
            W.append(w)
            D.append(d)
            assert w != 0.0 and d != 0.0
            feature = record['feature']
            assert not np.isnan(feature).flatten().sum()
            X.append(w/d)
            assert not np.isnan(X).flatten().sum()
            Y.append(feature)
            contours.append(record['cnt'])

    N = len(X)
    X = np.array(X)
    Y = np.array(Y)
    print('nan count: ', np.isnan(X).flatten().sum())
    print('nan count: ', np.isnan(Y).flatten().sum())
    print('inf count: ', np.isinf(X).flatten().sum())
    print('inf count: ', np.isinf(Y).flatten().sum())

    print('starting training model')
    K = 12
    X = np.reshape(X, (-1, 1))

    MODEL_PATH = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/models/hip.pkl'

    net = RBFNet(n_cluster=K, n_output=10, no_regress_at_ouputs=[0, 7])
    net.fit(X, Y)
    net.save_to_path(MODEL_PATH)

    net_1 = RBFNet.load_from_path(MODEL_PATH)

    OUTPUT_DEBUG_DIR_TRAIN = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/hip_prediction/train/'
    OUTPUT_DEBUG_DIR_TEST = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/hip_prediction/test/'
    shutil.rmtree(OUTPUT_DEBUG_DIR_TEST)
    shutil.rmtree(OUTPUT_DEBUG_DIR_TRAIN)
    os.makedirs(OUTPUT_DEBUG_DIR_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_DEBUG_DIR_TEST, exist_ok=True)

    for i in range(len(net.test_idxs)):
        idx = net_1.test_idxs[i]
        print('processing test idx: ', idx)
        pred = net_1.predict(np.expand_dims(X[idx, :], axis=0))[0, :]

        w = W[idx]
        d = D[idx]
        res_contour = util.reconstruct_slice_contour(pred, d, w, mirror=True)
        contour = contours[idx]
        center = util.contour_center(contour[0, :], contour[1, :])
        res_contour[0, :] += center[0]
        res_contour[1, :] += center[1]
        last_p = res_contour[:,0].reshape(2,1)
        res_contour = np.concatenate([res_contour, last_p], axis=1)
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(contour[0, :], contour[1, :], '-b')
        plt.plot(res_contour[0, :], res_contour[1, :], '-r')
        plt.plot(res_contour[0, :], res_contour[1, :], '+r')
        #plt.savefig(f'{OUTPUT_DEBUG_DIR_TEST}{idx}.png')
        plt.show()

    for i in range(len(net.train_idxs)):
        idx = net.train_idxs[i]
        print('processing train idx: ', idx)
        w = W[idx]
        d = D[idx]
        res_contour = util.reconstruct_slice_contour(pred, d, w)
        contour = contours[idx]
        center = util.contour_center(contour[0, :], contour[1, :])
        res_contour[0, :] += center[0]
        res_contour[1, :] += center[1]

        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(contour[0, :], contour[1, :], '-b')
        plt.plot(res_contour[0, :], res_contour[1, :], '-r')
        plt.savefig(f'{OUTPUT_DEBUG_DIR_TRAIN}{idx}.png')

