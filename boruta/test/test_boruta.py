import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

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
    X[:, 1] = (y * np.abs(np.random.normal(0, 1, 1000)) +
               np.random.normal(0, 0.1, 1000))
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


def test_dataframe_is_returned(Xy):
    X, y = Xy
    X_df, y_df = pd.DataFrame(X), pd.Series(y)
    rfc = RandomForestClassifier()
    bt = BorutaPy(rfc)
    bt.fit(X_df, y_df)
    assert isinstance(bt.transform(X_df, return_df=True), pd.DataFrame)


def test_selector_mixin_get_support_requires_fit():
    bt = BorutaPy(RandomForestClassifier())
    with pytest.raises(NotFittedError):
        bt.get_support()


def test_selector_mixin_get_support_matches_mask(Xy):
    X, y = Xy
    bt = BorutaPy(RandomForestClassifier())
    bt.fit(X, y)

    assert np.array_equal(bt.get_support(), bt.support_)
    assert np.array_equal(bt.get_support(indices=True),
                          np.where(bt.support_)[0])


def test_selector_mixin_inverse_transform_restores_selected_features(Xy):
    X, y = Xy
    bt = BorutaPy(RandomForestClassifier())
    bt.fit(X, y)

    X_selected = bt.transform(X)
    X_reconstructed = bt.inverse_transform(X_selected)

    assert X_reconstructed.shape == X.shape
    assert np.allclose(X_reconstructed[:, bt.support_], X[:, bt.support_])

    if (~bt.support_).any():
        assert np.allclose(X_reconstructed[:, ~bt.support_], 0)


def test_selector_mixin_get_feature_names_out_requires_fit():
    bt = BorutaPy(RandomForestClassifier())
    with pytest.raises(NotFittedError):
        bt.get_feature_names_out()


def test_selector_mixin_get_feature_names_out_returns_selected_names(Xy):
    X, y = Xy
    bt = BorutaPy(RandomForestClassifier())
    bt.fit(X, y)

    expected_default = np.array([f"x{i}" for i in np.where(bt.support_)[0]])
    assert np.array_equal(bt.get_feature_names_out(), expected_default)

    custom_names = np.array([f"feature_{i}" for i in range(X.shape[1])])
    selected_names = bt.get_feature_names_out(custom_names)
    assert np.array_equal(selected_names, custom_names[bt.support_])

    columns = [f"col_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    bt_df = BorutaPy(RandomForestClassifier())
    bt_df.fit(X_df, y)
    assert np.array_equal(bt_df.get_feature_names_out(), np.array(columns)[bt_df.support_])


@pytest.mark.parametrize("tree", [ExtraTreeClassifier(), DecisionTreeClassifier()])
def test_boruta_with_decision_trees(tree, Xy):
    msg = (
        f"The estimator {tree} does not take the parameter "
        "n_estimators. Use Random Forests or gradient boosting machines "
        "instead."
    )
    X, y = Xy
    bt = BorutaPy(tree)
    with pytest.raises(ValueError) as record:
        bt.fit(X, y)

    assert str(record.value) == msg
