from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

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