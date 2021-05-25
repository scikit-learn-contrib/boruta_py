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

    def __init__(self, estimator):
        self.estimator = estimator
        self.explainer = None
        self.feature_importances_ = None

    def set_params(self, n_estimators=1000, random_state=None):
        self.feature_importances_ = None
        self.estimator.set_params(n_estimators=n_estimators, random_state=random_state)

    def get_params(self):
        return self.estimator.get_params()

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.explainer = shap.TreeExplainer(self.estimator)
        self.feature_importances_ = np.absolute(self.explainer.shap_values(X)).sum(axis=0)


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

    def test_xgboost_version(self):
        np.random.seed(42)
        X, y = create_data()

        bst = xgb.XGBRFRegressor(tree_method="hist", max_depth=5, n_estimators=10)
        bt = BorutaPy(bst, n_estimators=bst.n_estimators)
        bt.fit(X, y)

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X)
        self.assertEqual(shap_values.shape, X.shape)

    def test_xgboost_shapley(self):
        np.random.seed(42)

        #  training data
        X, y = create_data()
        C = X[:, :3]  # features that are known to be important
        T = X[:, 3:]  # features to test -- only the first two in T should turn out to be important

        # Learner
        bst = Learner(xgb.XGBRFRegressor(**xgboost_parameters, n_estimators=10))

        # Boruta
        bt = BorutaPy(bst, n_estimators=bst.get_params()['n_estimators'])
        bt.fit(T, y, C)

        self.assertListEqual(list(range(2)), list(np.where(bt.support_)[0]))


if __name__ == '__main__':
    unittest.main()
