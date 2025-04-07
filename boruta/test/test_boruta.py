import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy


@pytest.mark.parametrize("tree_n,expected", [(10, 44), (100, 141)])
def test_get_tree_num(tree_n, expected):
    rfc = RandomForestClassifier(max_depth=10)
    bt = BorutaPy(rfc)
    assert bt._get_tree_num(tree_n) == expected


@pytest.fixture(scope="module")
def Xy():
    np.random.seed(42)
    y = np.random.binomial(1, 0.5, 1000)
    X = np.zeros((1000, 10))

    z = (y - np.random.binomial(1, 0.1, 1000) +
         np.random.binomial(1, 0.1, 1000))
    z[z == -1] = 0
    z[z == 2] = 1

    # 5 relevant features
    X[:, 0] = z
    X[:, 1] = (y * np.abs(np.random.normal(0, 1, 1000))
               + np.random.normal(0, 0.1, 1000))
    X[:, 2] = y + np.random.normal(0, 1, 1000)
    X[:, 3] = y**2 + np.random.normal(0, 1, 1000)
    X[:, 4] = np.sqrt(y) + np.random.binomial(2, 0.1, 1000)

    # 5 irrelevant features
    X[:, 5] = np.random.normal(0, 1, 1000)
    X[:, 6] = np.random.poisson(1, 1000)
    X[:, 7] = np.random.binomial(1, 0.3, 1000)
    X[:, 8] = np.random.normal(0, 1, 1000)
    X[:, 9] = np.random.poisson(1, 1000)

    return X, y


def test_if_boruta_extracts_relevant_features(Xy):
    X, y = Xy
    rfc = RandomForestClassifier()
    bt = BorutaPy(rfc)
    bt.fit(X, y)
    assert list(range(5)) == list(np.where(bt.support_)[0])


def test_if_it_works_with_dataframe_input(Xy):
    X, y = Xy
    X_df, y_df = pd.DataFrame(X), pd.Series(y)
    bt = BorutaPy(RandomForestClassifier())
    bt.fit(X_df, y_df)
    assert list(range(5)) == list(np.where(bt.support_)[0])


def test_it_dataframe_is_returned(Xy):
    X, y = Xy
    X_df, y_df = pd.DataFrame(X), pd.Series(y)
    rfc = RandomForestClassifier()
    bt = BorutaPy(rfc)
    bt.fit(X_df, y_df)
    assert isinstance(bt.transform(X_df, return_df=True), pd.DataFrame)
