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

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

def transform_data(n_cluster, X, use_sklean = False):
    if use_sklean:
        X = np.reshape(X, (-1, 1))
        kmeans_cls = KMeans(n_clusters=n_cluster, max_iter=1000, tol=1e-6).fit(X)
        X_labels = kmeans_cls.predict(X)

        #calculate standard variance
        cluster_std = np.zeros(len(kmeans_cls.labels_), dtype=np.float32)
        cluster_mean = kmeans_cls.cluster_centers_
        cluster_mean = cluster_mean.flatten()
        for l in kmeans_cls.labels_:
            cluster_mask = (X_labels == l)
            points = X[cluster_mask].flatten()
            cluster_std[l] = np.std(points)
    else:
        cluster_mean, cluster_std = kmeans(X, n_cluster)

    X_1 = []
    for i in range(X.shape[0]):
        a = np.array([rbf(X[i], c, s)[0] for c, s, in zip(cluster_mean, cluster_std)])
        X_1.append(a)
    X_1 = np.array(X_1)

    return X_1

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, out_shape = 1, lr=0.01, epochs=10, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

        input = Input(shape=(k,))
        output = Dense(out_shape)(input)
        self.model = Model(inputs = input, outputs = output)
        self.model.compile(optimizer=RMSprop(lr=0.001), loss= 'mean_squared_error', metrics=['mse'])

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        # training
        if True:
            X_1 = []
            for i in range(X.shape[0]):
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                X_1.append(a)
            X_1 = np.array(X_1)
        else:
            X_1 = transform_data(self.k, X, use_sklean=False)

        batch_size = 8
        epochs = 500
        model_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/female_slice_radial_code/hip_best_weight.hdf5'
        check_point = ModelCheckpoint(model_path, save_best_only=True)

        history = self.model.fit(x=X_1, y=y, batch_size=batch_size, epochs=epochs, callbacks=[check_point], validation_split=0.2)
        plt.plot(np.arange(epochs), history.history['loss'], '-r')
        plt.plot(np.arange(epochs), history.history['val_loss'], '-b')
        plt.show()

        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    def predict_1(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            a = np.expand_dims(a, axis=0)
            F = self.model.predict(a)
            y_pred.append(F.flatten())
        return np.array(y_pred)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)


def split_data(K, X_1, Y):
    N = X_1.shape[0]

    X_train, X_test, train_idxs, test_idxs = train_test_split(X_1, np.arange(N), test_size=0.1, shuffle=True)


    #print(f'y train min, max = {Y_train.min()}, {Y_train.max()}')
    X_train, X_valid, train_idxs, valid_idxs = train_test_split(X_train, train_idxs, test_size=0.05, shuffle=True)

    Y_train = Y[train_idxs, :]
    Y_valid = Y[valid_idxs, :]
    Y_test =  Y[test_idxs, :]

    #Y_train_mean = np.mean(Y_train, axis=0)
    #Y_train_std  = np.std(Y_train, axis=0)
    #Y_train = (Y_train - Y_train_mean) / Y_train_std
    #Y_valid  = (Y_valid - Y_train_mean) / Y_train_std
    #Y_test  = (Y_test  - Y_train_mean) / Y_train_std

    return X_1, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, train_idxs, valid_idxs, test_idxs

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

def test_rbf_network():
    # sample inputs and add noise
    # sample inputs and add noise
    NUM_SAMPLES = 100
    X = np.random.uniform(0., 1., NUM_SAMPLES)
    X = np.sort(X, axis=0)
    noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
    y = np.sin(2 * np.pi * X) + noise

    rbfnet = RBFNet(lr=1e-2, k=2)
    rbfnet.fit(X, y)

    y_pred = rbfnet.predict(X)
    y_pred_1 = rbfnet.predict_1(X)

    plt.plot(X, y, '-o', label='true')
    plt.plot(X, y_pred, '-o', label='RBF-Net')
    plt.plot(X, y_pred_1, '-r', label='RBF-Net')
    plt.legend()

    plt.tight_layout()
    plt.show()

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
    X_1 = transform_data(K, X, use_sklean=True)

    X_1, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, train_idxs, valid_idxs, test_idxs = split_data(K, X_1, Y)

    if False:
        input = Input(shape=(K,))
        output = Dense(10)(input)
        model = Model(inputs = input, outputs = output)
        #model.compile(optimizer=RMSprop(lr=0.1), loss= 'mean_squared_error', metrics=['mse'])
        model.compile(optimizer=Adadelta(), loss= 'mean_squared_error', metrics=['mse'])

        batch_size = 8
        epochs = 400
        model_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/female_slice_radial_code/hip_best_weight.hdf5'
        check_point = ModelCheckpoint(model_path, save_best_only=True)
        history = model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), batch_size=batch_size, epochs=epochs, callbacks=[check_point])
        for i in range(5):
            print(model.predict(np.expand_dims(X_test[i, :], axis=0)))
            print(Y_test[i,:])
        plt.plot(np.arange(epochs), history.history['loss'], '-r')
        plt.plot(np.arange(epochs), history.history['val_loss'], '-b')
        plt.show()
    else:
        #reg = LinearRegression().fit(X_train, Y_train)
        reg = ExtraTreesRegressor().fit(X_train, Y_train)
        print('linear regression score: ', reg.score(X_train, Y_train))
        print('linear regression score: ', reg.score(X_valid, Y_valid))

        OUTPUT_DEBUG_DIR_TRAIN = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/hip_prediction/train/'
        OUTPUT_DEBUG_DIR_TEST = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/debug/hip_prediction/test/'
        shutil.rmtree(OUTPUT_DEBUG_DIR_TEST)
        shutil.rmtree(OUTPUT_DEBUG_DIR_TRAIN)
        os.makedirs(OUTPUT_DEBUG_DIR_TRAIN, exist_ok=True)
        os.makedirs(OUTPUT_DEBUG_DIR_TEST, exist_ok=True)

        for i in range(len(test_idxs)):
            idx = test_idxs[i]
            print('progress: ', idx)
            pred = reg.predict(np.expand_dims(X_1[idx, :], axis=0))[0,:]
            w = W[idx]
            d = D[idx]
            res_contour = util.reconstruct_slice_contour(pred, d, w)
            contour = contours[idx]
            center  = util.contour_center(contour[0,:], contour[1,:])
            res_contour[0,:] += center[0]
            res_contour[1,:] += center[1]

            plt.clf()
            plt.axes().set_aspect(1)
            plt.plot(contour[0,:], contour[1,:], '-b')
            plt.plot(res_contour[0,:], res_contour[1,:], '-r')
            plt.savefig(f'{OUTPUT_DEBUG_DIR_TEST}{idx}.png')


        for i in range(len(train_idxs)):
            idx = train_idxs[i]
            print('progress: ', idx)
            pred = reg.predict(np.expand_dims(X_1[idx, :], axis=0))[0,:]
            w = W[idx]
            d = D[idx]
            res_contour = util.reconstruct_slice_contour(pred, d, w)
            contour = contours[idx]
            center  = util.contour_center(contour[0,:], contour[1,:])
            res_contour[0,:] += center[0]
            res_contour[1,:] += center[1]

            plt.clf()
            plt.axes().set_aspect(1)
            plt.plot(contour[0,:], contour[1,:], '-b')
            plt.plot(res_contour[0,:], res_contour[1,:], '-r')
            plt.savefig(f'{OUTPUT_DEBUG_DIR_TRAIN}{idx}.png')
