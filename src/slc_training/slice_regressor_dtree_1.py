from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import pickle
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from dtreeviz.trees import *
import sys

class GaussianMixture_Hack(GaussianMixture):

    def fit_transform(self, X, y, **fit_params):
        self.fit(X)
        return self.predict_proba(X)

    def transform(self, X):
        return self.predict_proba(X)

class SliceRegressor():
    def __init__(self, slc_id, model_input_slc_ids):
        self._slc_id = slc_id
        self._model_slc_input_ids = model_input_slc_ids

    def save_to_path(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @property
    def slc_id(self):
        return self._slc_id

    @property
    def slc_model_input_ids(self):
        return self._model_slc_input_ids

    def fit(self, X, Y, n_jobs = -1):
        N = X.shape[0]
        X_train, X_test, train_idxs, test_idxs = train_test_split(X, np.arange(N), test_size=0.2, shuffle=False, random_state=200)
        train_idxs = train_idxs
        test_idxs = test_idxs
        Y_train = Y[train_idxs]
        Y_test = Y[test_idxs]

        print('searching for best optimator')
        #regressor = ExtraTreesRegressor(random_state=200)
        #regressor = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50))
        regressor = Pipeline([('gmm', GaussianMixture_Hack(n_init=10)), ('reg', ExtraTreesRegressor())])
        parameters = {'gmm__n_components':np.arange(10,16,step=2), 'reg__n_estimators':np.arange(80,160,step=20), 'reg__min_samples_leaf':np.arange(20,140,step=20)}
        print(f'\tsearch parameters: {parameters}')
        clf = GridSearchCV(regressor, parameters, cv = 5, n_jobs=n_jobs)
        clf.fit(X_train, Y_train)
        print(clf.best_estimator_)
        self.regressor = clf.best_estimator_

        test_score = self.regressor.score(X_test, Y_test)
        train_score = self.regressor.score(X_train, Y_train)
        print('\tregression score on train set:  ', train_score)
        print('\tregression score on test set: ', test_score)
        print('\tregression score on test mse loss: ', self.loss(X_test, Y_test))

        return train_idxs, test_idxs, train_score, test_score

    def loss(self, X, Y):
        Y_hat = self.regressor.predict(X)
        l = np.sqrt(np.mean((Y-Y_hat)**2))
        return l

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        assert len(X.shape) == 2

        preds = self.regressor.predict(X)
        return preds


class SliceRegressorLocalGlobal():
    def __init__(self, slc_id, model_input_slc_ids, model_global_input_slc_ids):
        self._slc_id = slc_id
        self._model_slc_local_input_ids = model_input_slc_ids
        self._model_slc_global_input_ids = model_global_input_slc_ids

    def save_to_path(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @property
    def slc_id(self):
        return self._slc_id

    @property
    def slc_model_input_ids(self):
        return self._model_slc_local_input_ids + self._model_slc_global_input_ids

    def local_column(self, X):
        n_local = len(self._model_slc_local_input_ids)
        return X[:,:n_local]

    def global_column(self, X):
        n_local = len(self._model_slc_local_input_ids)
        return X[:, n_local:]

    def fit(self, X, Y, n_jobs = -1):
        assert len(X.shape) == 2
        assert X.shape[1] == len(self.slc_model_input_ids)

        N = X.shape[0]

        X_train, X_test, train_idxs, test_idxs = train_test_split(X, np.arange(N), test_size=0.2, shuffle=False, random_state=200)
        train_idxs = train_idxs
        test_idxs = test_idxs
        Y_train = Y[train_idxs]
        Y_test = Y[test_idxs]

        local_data = FunctionTransformer(self.local_column,  validate=False)
        global_data = FunctionTransformer(self.global_column, validate=False)


        print('searching for best optimator')
        #regressor = ExtraTreesRegressor(random_state=200)
        #regressor = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50))
        regressor = Pipeline([('features', FeatureUnion([
                                            ('local_feature', Pipeline([('local_data',  local_data), ('gmm', GaussianMixture_Hack(n_init=10))])),
                                            ('global_feature',Pipeline([('global_data',global_data), ('gmm_global', GaussianMixture_Hack(n_init=20))]))]))
                             ,('reg', ExtraTreesRegressor())])

        print(regressor.named_steps)
        parameters = {'features__local_feature__gmm__n_components':np.arange(10,16,step=2),
                      'reg__n_estimators':np.arange(80,160,step=20), 'reg__min_samples_leaf':np.arange(20,140,step=20)}



        print(f'\tsearch parameters: {parameters}')
        clf = GridSearchCV(regressor, parameters, cv = 5, n_jobs=n_jobs)
        clf.fit(X_train, Y_train)
        print(clf.best_estimator_)
        self.regressor = clf.best_estimator_

        train_score = self.regressor.score(X_train, Y_train)
        test_score = self.regressor.score(X_test, Y_test)
        print('\tregression score on train set:  ', train_score)
        print('\tregression score on test set: ', test_score)
        print('\tregression score on test mse loss: ', self.loss(X_test, Y_test))

        return train_idxs, test_idxs, train_score, test_score

    def loss(self, X, Y):
        Y_hat = self.regressor.predict(X)
        l = np.sqrt(np.mean((Y-Y_hat)**2))
        return l

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        assert len(X.shape) == 2

        preds = self.regressor.predict(X)
        return preds


