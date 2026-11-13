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


@pytest.mark.parametrize("n_features", [1, 3])
@pytest.mark.parametrize("max_iter,confirmed", [(6, False), (20, True)])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_no_rejected_features_preserves_confirmation_status(
        n_features, max_iter, confirmed, method):
    y = np.tile([0, 1], 100)
    X = np.tile(y[:, np.newaxis], (1, n_features))
    bt = BorutaPy(RandomForestClassifier(max_depth=3), n_estimators=32,
                  random_state=7, max_iter=max_iter)

    result = getattr(bt, method)(X, y)

    # The short run cannot confirm the features, but their importance is
    # sufficient to keep all of them as tentative after the rough fix.
    assert bt.n_features_ == (n_features if confirmed else 0)
    np.testing.assert_array_equal(bt.ranking_, np.full(n_features, 1 if confirmed else 2))
    np.testing.assert_array_equal(bt.support_weak_, np.full(n_features, not confirmed))
    np.testing.assert_array_equal(bt.support_, np.full(n_features, confirmed))
    assert not np.any(bt.support_ & bt.support_weak_)
    np.testing.assert_array_equal(bt.get_support(), bt.support_)
    assert bt.get_feature_names_out().size == bt.n_features_
    assert bt.transform(X).shape == (len(y), bt.n_features_)
    np.testing.assert_array_equal(bt.transform(X, weak=True), X)
    if method == "fit_transform":
        np.testing.assert_array_equal(result, bt.transform(X))
    else:
        assert result is bt


def test_no_rejected_features_preserves_mixed_confirmation_status():
    y = np.tile([0, 1], 100)
    rng = np.random.RandomState(0)
    noisy_y = np.logical_xor(y, rng.random_sample(y.size) < 0.4).astype(int)
    X = np.column_stack((y, noisy_y))
    bt = BorutaPy(RandomForestClassifier(max_depth=3), n_estimators=32,
                  random_state=7, max_iter=9).fit(X, y)

    assert bt.n_features_ == 1
    np.testing.assert_array_equal(bt.ranking_, [1, 2])
    np.testing.assert_array_equal(bt.support_weak_, [False, True])
    np.testing.assert_array_equal(bt.support_, [True, False])
    np.testing.assert_array_equal(bt.get_support(indices=True), [0])
    np.testing.assert_array_equal(bt.get_feature_names_out(), ["x0"])
    np.testing.assert_array_equal(bt.transform(X), X[:, :1])
    np.testing.assert_array_equal(bt.transform(X, weak=True), X)


def test_all_features_rejected_preserves_empty_support():
    y = np.tile([0, 1], 100)
    X = np.zeros((y.size, 3))
    bt = BorutaPy(RandomForestClassifier(max_depth=3), n_estimators=32,
                  random_state=7, max_iter=20).fit(X, y)

    assert bt.n_features_ == 0
    assert not bt.support_.any()
    assert not bt.support_weak_.any()
    assert bt.transform(X).shape == (y.size, 0)
    assert bt.transform(X, weak=True).shape == (y.size, 0)
