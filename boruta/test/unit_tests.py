import unittest
from boruta import BorutaPy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import shap
import xgboost as xgb

xgboost_parameters = {
    "alpha": 0.0,
    "colsample_bylevel": 1.0,
    "colsample_bytree": 1.0,
    "eta": 0.3,
    "eval_metric": ["error"],
    "gamma": 0.0,
    "lambda": 1.0,
    "max_bin": 256,
    "max_delta_step": 0,
    "max_depth": 6,
    "min_child_weight": 1,
    "nthread": -1,
    "objective": "binary:logistic",
    "subsample": 1.0,
    "tree_method": "auto"
}


class Learner:

    def __init__(self, params, nrounds=1000, verbose=False):
        self.params = params
        self.nrounds = nrounds
        self.feature_importances_ = None
        self.verbose = verbose

    def set_params(self, n_estimators=None, random_state=None):
        """
        used by boruta_py but essentially useless in the case of xgboost.
        :param n_estimators: the number of rounds, typically hard set in xgboost
        :param random_state: ignored
        """
        self.feature_importances_ = None
        if n_estimators:
            self.nrounds = n_estimators

    def get_params(self):
        return self.params

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        eval_set = [(dtrain, 'test')]
        model = xgb.train(self.params, dtrain, num_boost_round=self.nrounds, evals=eval_set, verbose_eval=self.verbose)
        explainer = shap.TreeExplainer(model)
        self.feature_importances_ = np.absolute(explainer.shap_values(X)).sum(axis=0)


def create_data():
    y = np.random.binomial(1, 0.5, 1000)
    X = np.zeros((1000, 10))

    z = y - np.random.binomial(1, 0.1, 1000) + np.random.binomial(1, 0.1, 1000)
    z[z == -1] = 0
    z[z == 2] = 1

    # 5 relevant features
    X[:, 0] = z
    X[:, 1] = y * np.abs(np.random.normal(0, 1, 1000)) + np.random.normal(0, 0.1, 1000)
    X[:, 2] = y + np.random.normal(0, 1, 1000)
    X[:, 3] = y ** 2 + np.random.normal(0, 1, 1000)
    X[:, 4] = np.sqrt(y) + np.random.binomial(2, 0.1, 1000)

    # 5 irrelevant features
    X[:, 5] = np.random.normal(0, 1, 1000)
    X[:, 6] = np.random.poisson(1, 1000)
    X[:, 7] = np.random.binomial(1, 0.3, 1000)
    X[:, 8] = np.random.normal(0, 1, 1000)
    X[:, 9] = np.random.poisson(1, 1000)
    return X, y


class BorutaTestCases(unittest.TestCase):

    def test_get_tree_num(self):
        rfc = RandomForestClassifier(max_depth=10)
        bt = BorutaPy(rfc)
        self.assertEqual(bt._get_tree_num(10), 44, "Tree Est. Math Fail")
        self.assertEqual(bt._get_tree_num(100), 141, "Tree Est. Math Fail")

    def test_if_boruta_extracts_relevant_features(self):
        np.random.seed(42)
        X, y = create_data()

        rfc = RandomForestClassifier()
        bt = BorutaPy(rfc)
        bt.fit(X, y)

        # make sure that only all the relevant features are returned
        self.assertListEqual(list(range(5)), list(np.where(bt.support_)[0]))

        # test if this works as expected for dataframe input
        X_df, y_df = pd.DataFrame(X), pd.Series(y)
        bt.fit(X_df, y_df)
        self.assertListEqual(list(range(5)), list(np.where(bt.support_)[0]))

        # check it dataframe is returned when return_df=True
        self.assertIsInstance(bt.transform(X_df, return_df=True), pd.DataFrame)

    def test_xgboost_all_features(self):
        np.random.seed(42)
        X, y = create_data()

        bst = Learner(xgboost_parameters, nrounds=10)
        bt = BorutaPy(bst, n_estimators=bst.nrounds, verbose=True)
        bt.fit(X, y)

        # make sure that only all the relevant features are returned
        self.assertListEqual(list(range(5)), list(np.where(bt.support_)[0]))

    def test_xgboost_some_features(self):
        np.random.seed(42)

        #  training data
        X, y = create_data()
        C = X[:, :3]  # features that are known to be important
        T = X[:, 3:]  # features to test -- only the first two in T should turn out to be important

        # Learner
        bst = Learner(xgboost_parameters, nrounds=25)

        # Boruta
        bt = BorutaPy(bst, n_estimators=bst.nrounds, max_iter=10, verbose=True)
        bt.fit(T, y, C)

        self.assertListEqual(list(range(2)), list(np.where(bt.support_)[0]))


if __name__ == '__main__':
    unittest.main()
