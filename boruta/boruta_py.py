#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

from __future__ import print_function, division
import numpy as np
import scipy as sp
import shap
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.inspection import permutation_importance
from matplotlib.lines import Line2D
import warnings


class BorutaPy(BaseEstimator, TransformerMixin):
    """
    Improved Python implementation of the Boruta R package.
    The improvements of this implementation include:
    - Faster run times:
        Thanks to scikit-learn's fast implementation of the ensemble methods.
    - Scikit-learn like interface:
        Use BorutaPy just like any other scikit learner: fit, fit_transform and
        transform are all implemented in a similar fashion.
    - Modularity:
        Any ensemble method could be used: random forest, extra trees
        classifier, even gradient boosted trees.
    - Two step correction:
        The original Boruta code corrects for multiple testing in an overly
        conservative way. In this implementation, the Benjamini Hochberg FDR is
        used to correct in each iteration across active features. This means
        only those features are included in the correction which are still in
        the selection process. Following this, each that passed goes through a
        regular Bonferroni correction to check for the repeated testing over
        the iterations.
    - Percentile:
        Instead of using the max values of the shadow features the user can
        specify which percentile to use. This gives a finer control over this
        crucial parameter. For more info, please read about the perc parameter.
    - Automatic tree number:
        Setting the n_estimator to 'auto' will calculate the number of trees
        in each itartion based on the number of features under investigation.
        This way more trees are used when the training data has many feautres
        and less when most of the features have been rejected.
    - Ranking of features:
        After fitting BorutaPy it provides the user with ranking of features.
        Confirmed ones are 1, Tentatives are 2, and the rejected are ranked
        starting from 3, based on their feature importance history through
        the iterations.
    - Using either the native variable importance, scikit permutation importance,
        SHAP importance.
    We highly recommend using pruned trees with a depth between 3-7.
    For more, see the docs of these functions, and the examples below.
    Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
    Boruta is an all relevant feature selection method, while most other are
    minimal optimal; this means it tries to find all features carrying
    information usable for prediction, rather than finding a possibly compact
    subset of features on which some classifier has a minimal error.
    Why bother with all relevant feature selection?
    When you try to understand the phenomenon that made your data, you should
    care about all factors that contribute to it, not just the bluntest signs
    of it in context of your methodology (yes, minimal optimal set of features
    by definition depends on your classifier choice).
    Parameters
    ----------
    estimator : object
        A supervised learning estimator, with a 'fit' method that returns the
        feature_importances_ attribute. Important features must correspond to
        high absolute values in the feature_importances_.
    n_estimators : int or string, default = 1000
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.
    perc : int, default = 100
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        The default is essentially the vanilla Boruta corresponding to the max.
    alpha : float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.
    importance : str, default = 'shap'
        The kind of variable importance used to compare and discriminate original
        vs shadow predictors.
    two_step : Boolean, default = True
        If you want to use the original implementation of Boruta with Bonferroni
        correction only set this to False.
    max_iter : int, default = 100
        The number of maximum iterations to perform.
    random_state : int, RandomState instance or None; default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    weight : pd.Series
    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already
    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features - only confirmed ones are True.
    support_weak_ : array of shape [n_features]
        The mask of selected tentative features, which haven't gained enough
        support during the max_iter number of iterations..
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1 and tentative features are assigned
        rank 2.
    Examples
    --------

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy

    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    y = y.ravel()

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)

    # check selected features - first 5 features are selected
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    References
    ----------
    [1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
        Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
    """

    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05, importance='shap',
                 two_step=True, max_iter=100, random_state=None, weight=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.weight = weight
        self.importance = importance
        self.cat_name = None
        self.cat_idx = None
        # plotting
        self.imp_real_hist = None
        self.sha_max = None
        self.col_names = None
        # Catboost doesn't allow to change random seed after fitting
        self._is_catboost = 'catboost' in str(type(self.estimator))
        # Random state throws an error with lightgbm
        self._is_lightgbm = 'lightgbm' in str(type(self.estimator))

    def fit(self, X, y):
        """
        Fits the Boruta feature selection with the provided estimator.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        self.imp_real_hist = np.empty((0, X.shape[1]), float)
        if isinstance(X, pd.DataFrame) is not True:
            X = pd.DataFrame(X)
        self.col_names = X.columns.to_list()
        return self._fit(X, y)

    def transform(self, X, weak=False, return_df=False):
        """
        Reduces the input X to the features selected by Boruta.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        return self._transform(X, weak, return_df)

    def fit_transform(self, X, y, weak=False, return_df=False):
        """
        Fits Boruta, then reduces the input X to the selected features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        self._fit(X, y)
        return self._transform(X, weak, return_df)

    def plot_importance(self):
        """
        Boxplot of the variable importance, ordered by magnitude
        The max shadow variable importance illustrated by the dashed line.
        Requires to apply the fit method first.
        :return: boxplot
        """
        if self.imp_real_hist is None:
            raise ValueError("Use the fit method first to compute the var.imp")

        color = {'boxes': 'gray', 'whiskers': 'gray', 'medians': '#404040', 'caps': 'gray'}
        vimp_df = pd.DataFrame(self.imp_real_hist, columns=self.col_names)
        vimp_df = vimp_df.reindex(vimp_df.mean().sort_values(ascending=True).index, axis=1)
        bp = vimp_df.boxplot(color=color,
                             boxprops=dict(linestyle='-', linewidth=1.5),
                             flierprops=dict(linestyle='-', linewidth=1.5),
                             medianprops=dict(linestyle='-', linewidth=1.5),
                             whiskerprops=dict(linestyle='-', linewidth=1.5),
                             capprops=dict(linestyle='-', linewidth=1.5),
                             showfliers=False, grid=True, rot=0, vert=False, patch_artist=True
                             )

        box_face_col = ["tab:blue"] * sum(self.support_) + ["#FDD023"] * sum(self.support_weak_)
        for c in range(len(box_face_col)):
            bp.findobj(matplotlib.patches.Patch)[len(self.support_) - c - 1].set_facecolor(box_face_col[c])
            bp.findobj(matplotlib.patches.Patch)[len(self.support_) - c - 1].set_facecolor(box_face_col[c])

        xrange = vimp_df.max(skipna=True).max(skipna=True)-vimp_df.min(skipna=True).min(skipna=True)
        bp.set_xlim(left=vimp_df.min(skipna=True).min(skipna=True)-0.10*xrange)
        custom_lines = [Line2D([0], [0], color="tab:blue", lw=5),
                        Line2D([0], [0], color="#FDD023", lw=5),
                        Line2D([0], [0], color="gray", lw=5),
                        Line2D([0], [0], linestyle='--', color="gray", lw=2)]
        bp.legend(custom_lines, ['confirmed', 'tentative', 'rejected', 'sha. max'], loc="lower right")
        plt.axvline(x=self.sha_max, linestyle='--', color='gray')
        fig = bp.get_figure()
        plt.title('Boruta importance and selected predictors')
        fig.set_size_inches((10, 2 * np.rint(max(vimp_df.shape) / 10)))
        plt.tight_layout()
        plt.show()

    def _is_tree_based(self):
        """
        checking if the estimator is tree-based (kernel SAP is too slow to be used here, unless using sampling)
        :return:
        """
        tree_based_models = ['lightgbm', 'xgboost', 'catboost', '_forest']
        condition = any(i in str(type(self.estimator)) for i in tree_based_models)
        return condition

    def _validate_pandas_input(self, arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    def _fit(self, X_raw, y):

        start_time = time.time()
        # basic cat features encoding
        # First, let's store "object" columns as categorical columns
        # obj_feat = X_raw.dtypes.loc[X_raw.dtypes == 'object'].index.tolist()
        obj_feat = list(set(list(X_raw.columns)) - set(list(X_raw.select_dtypes(include=[np.number]))))

        if obj_feat:
            X_raw.loc[:, obj_feat] = X_raw[obj_feat].astype('str').astype('category')
        cat_feat = X_raw.dtypes.loc[X_raw.dtypes == 'category'].index.tolist()
        cat_idx = [X_raw.columns.get_loc(c) for c in cat_feat if c in cat_feat]
        if cat_feat:
            # a way without loop but need to re-do astype
            Cat = X_raw[cat_feat].stack().astype('category').cat.codes.unstack()

        if not self._is_catboost:
            if cat_feat:
                X = pd.concat([X_raw[X_raw.columns.difference(cat_feat)], Cat], axis=1)
            else:
                X = X_raw
        else:
            X = X_raw

        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        # w = w.fillna(0)

        self.cat_name = cat_feat
        self.cat_idx = cat_idx

        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X)

        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            # Catboost doesn't allow to change random seed after fitting
            if not self._is_catboost:
                if self._is_lightgbm:
                    # https://github.com/scikit-learn-contrib/boruta_py/pull/78
                    self.estimator.set_params(random_state=self.random_state.randint(0, 10000))
                else:
                    self.estimator.set_params(random_state=self.random_state)


            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1
        # for plotting
        self.imp_real_hist = imp_history
        self.sha_max = imp_sha_max
        if isinstance(X_raw, pd.DataFrame):
            self.support_names_ = [X_raw.columns[i] for i, x in enumerate(self.support_) if x]

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)

            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        self.running_time = time.time() - start_time
        hours, rem = divmod(self.running_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("All relevant predictors selected in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        return self

    def _transform(self, X, weak=False, return_df=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        if return_df:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]
        return X

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if depth == None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        # n_feat * 2 because the training matrix is extended with n shadow features
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _get_imp(self, X, y):
        try:
            if self._is_catboost:
                X = pd.DataFrame(X)
                obj_feat = X.dtypes.loc[(X.dtypes == 'object') | (X.dtypes == 'category')].index.tolist()
                if obj_feat:
                    X[obj_feat] = X[obj_feat].astype('str').astype('category')
                cat_feat = X.dtypes.loc[X.dtypes == 'category'].index.tolist()
                self.estimator.fit(X, y, sample_weight=self.weight, cat_features=cat_feat)
            else:
                self.estimator.fit(X, y, sample_weight=self.weight)

        except Exception as e:
            raise ValueError('Please check your X and y variable. The provided '
                             'estimator cannot be fitted to your data.\n' + str(e))
        try:
            imp = self.estimator.feature_importances_
        except Exception:
            raise ValueError('Only methods with feature_importance_ attribute '
                             'are currently supported in BorutaPy.')
        return imp

    def _get_shap_imp(self, X, y):
        # SHAP and permutation importances must be computed on unseen data
        if self.weight is not None:
            w = self.weight
            if is_regressor(self.estimator):
                X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, random_state=42)
            else:
                X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, stratify=y, random_state=42)
        else:
            if is_regressor(self.estimator):
                X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, random_state=42)
            else:
                X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, stratify=y, random_state=42)
            w_tr, w_tt = None, None

        X_tr = pd.DataFrame(X_tr)
        X_tt = pd.DataFrame(X_tt)
        obj_feat = list(set(list(X_tr.columns)) - set(list(X_tr.select_dtypes(include=[np.number]))))
        obj_idx = None

        if obj_feat:
            X_tr[obj_feat] = X_tr[obj_feat].astype('str').astype('category')
            X_tt[obj_feat] = X_tt[obj_feat].astype('str').astype('category')
            obj_idx = np.argwhere(X_tr.columns.isin(obj_feat)).ravel()

        if self._is_tree_based():
            try:
                if self._is_catboost:
                    model = self.estimator.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=obj_feat)
                else:
                    model = self.estimator.fit(X_tr, y_tr, sample_weight=w_tr)

            except Exception as e:
                raise ValueError('Please check your X and y variable. The provided '
                                 'estimator cannot be fitted to your data.\n' + str(e))
            # build the explainer
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(X_tt)
            # flatten to 2D if classification and lightgbm
            if is_classifier(self.estimator):
                if isinstance(shap_values, list):
                    # for lightgbm clf sklearn api, shap returns list of arrays
                    # https://github.com/slundberg/shap/issues/526
                    class_inds = range(len(shap_values))
                    shap_imp = np.zeros(shap_values[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(shap_values[ind]).mean(0)
                    shap_imp /= len(shap_values)
                else:
                    shap_imp = np.abs(shap_values).mean(0)
            else:
                shap_imp = np.abs(shap_values).mean(0)
        else:
            raise ValueError('Not a tree based model')

        return shap_imp

    def _get_perm_imp(self, X, y):

        if self.weight is not None:
            w = self.weight
            if is_regressor(self.estimator):
                X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, random_state=42)
            else:
                X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, stratify=y, random_state=42)
        else:
            if is_regressor(self.estimator):
                X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, random_state=42)
            else:
                X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, stratify=y, random_state=42)
            w_tr, w_tt = None, None

        X_tr = pd.DataFrame(X_tr)
        X_tt = pd.DataFrame(X_tt)
        obj_feat = list(set(list(X_tr.columns)) - set(list(X_tr.select_dtypes(include=[np.number]))))
        obj_idx = None

        if obj_feat:
            X_tr[obj_feat] = X_tr[obj_feat].astype('str').astype('category')
            X_tt[obj_feat] = X_tt[obj_feat].astype('str').astype('category')
            obj_idx = np.argwhere(X_tr.columns.isin(obj_feat)).ravel()

        if self._is_tree_based():
            try:
                if self._is_catboost:
                    model = self.estimator.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=obj_feat)
                else:
                    model = self.estimator.fit(X_tr, y_tr, sample_weight=w_tr)

            except Exception as e:
                raise ValueError('Please check your X and y variable. The provided '
                                 'estimator cannot be fitted to your data.\n' + str(e))

            perm_imp = permutation_importance(model, X_tt, y_tt, n_repeats=5, random_state=42, n_jobs=-1)
            imp = perm_imp.importances_mean.ravel()
        else:
            raise ValueError('Not a tree based model')

        return imp

    def _get_shuffle(self, seq):
        self.random_state.shuffle(seq)
        return seq

    def _add_shadows_get_imps(self, X, y, dec_reg):
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = np.copy(x_cur)
        # make sure there's at least 5 columns in the shadow matrix for
        while (x_sha.shape[1] < 5):
            x_sha = np.hstack((x_sha, x_sha))
        # shuffle xSha
        x_sha = np.apply_along_axis(self._get_shuffle, 0, x_sha)
        # get importance of the merged matrix
        if self.importance == 'shap':
            imp = self._get_shap_imp(np.hstack((x_cur, x_sha)), y)
        elif self.importance == 'pimp':
            imp = self._get_perm_imp(np.hstack((x_cur, x_sha)), y)
        else:
            imp = self._get_imp(np.hstack((x_cur, x_sha)), y)

        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]

        return imp_real, imp_sha

    def _assign_hits(self, hit_reg, cur_imp, imp_sha_max):
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp[0]
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = self._fdrcorrection(to_accept_ps, alpha=self.alpha)[0]
            to_reject = self._fdrcorrection(to_reject_ps, alpha=self.alpha)[0]

            # second we correct for testing the same feature over and over again
            # using bonferroni
            to_accept2 = to_accept_ps <= self.alpha / float(_iter)
            to_reject2 = to_reject_ps <= self.alpha / float(_iter)

            # combine the two multi corrections, and get indexes
            to_accept *= to_accept2
            to_reject *= to_reject2
        else:
            # as in th original Boruta, we simply do bonferroni correction
            # with the total n_feat in each iteration
            to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
            to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    def _fdrcorrection(self, pvals, alpha=0.05):
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.
        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate
        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_

    def _nanrankdata(self, X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y, dtype=None, force_all_finite=False)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _print_results(self, dec_reg, _iter, flag):
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            if self.importance in ['shap', 'pimp']:
                vimp = str(self.importance)
            else:
                vimp = 'native'
            output = "\n\nBorutaPy finished running using " + vimp + " var. imp.\n\n" + result
        print(output)
