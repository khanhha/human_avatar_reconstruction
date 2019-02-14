from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from dtreeviz.trees import *
import sys

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

    def save_to_path(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def fit(self, X, Y):
        self.cluster_data(X)
        X_1 = self.transform_data(X)

        nan_mask = np.isnan(X_1)
        nan_mask = np.sum(nan_mask, axis=1) > 0
        print(f'\ttransformed_data: nan count: {np.sum(nan_mask[:])}')
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
        print('\tregression score on train set:  ', self.regressor.score(X_train, Y_train))
        print('\tregression score on test set: ', self.regressor.score(X_test, Y_test))
        print('\tregression score on test mse loss: ', self.loss(X_test, Y_test))

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
                    viz.save(f'{DIR}/{self.slc_id}/{i}_{j}.svg')

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



