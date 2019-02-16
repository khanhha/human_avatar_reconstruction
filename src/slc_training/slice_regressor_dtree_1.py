from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import pickle
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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

    def fit(self, X, Y):
        N = X.shape[0]
        X_train, X_test, train_idxs, test_idxs = train_test_split(X, np.arange(N), test_size=0.2, shuffle=True, random_state=200)
        train_idxs = train_idxs
        test_idxs = test_idxs
        Y_train = Y[train_idxs]
        Y_test = Y[test_idxs]

        print('searching for best optimator')
        #regressor = ExtraTreesRegressor(random_state=200)
        #regressor = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50))
        regressor = Pipeline([('gmm', GaussianMixture_Hack(n_init=10)), ('reg', ExtraTreesRegressor())])
        parameters = {'gmm__n_components':np.arange(8,18,step=2), 'reg__n_estimators':np.arange(40,140,step=20), 'reg__min_samples_leaf':np.arange(20,140,step=20)}
        print(f'\tsearch parameters: {parameters}')
        clf = GridSearchCV(regressor, parameters, cv = 5, verbose=1 , n_jobs=-1)
        clf.fit(X_train, Y_train)
        print(clf.best_estimator_)
        self.regressor = clf.best_estimator_

        print('\tregression score on train set:  ', self.regressor.score(X_train, Y_train))
        print('\tregression score on test set: ', self.regressor.score(X_test, Y_test))
        print('\tregression score on test mse loss: ', self.loss(X_test, Y_test))


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




