"""
This module provides a method to perform 'all relevant feature selection'


Reference:
----------
NILSSON, Roland, PEÑA, José M., BJÖRKEGREN, Johan, et al.
Consistent feature selection for pattern recognition in polynomial time.
Journal of Machine Learning Research, 2007, vol. 8, no Mar, p. 589-612.

KURSA, Miron B., RUDNICKI, Witold R., et al.
Feature selection with the Boruta package.
J Stat Softw, 2010, vol. 36, no 11, p. 1-13.


The module structure is the following:
---------------------------------------
- The ``BorutaPy`` class, the main class, with a scikit-learn like syntax. It performs all relevant FS
  as described in the original paper but implements several additional features, as native handling
  of categorical predictors, different feature importance (native, pimp and shap,
  if the SHAP pkg is installed). It works with CatBoost scikit-learn models and you can pass
  sample_weight.
"""

from __future__ import print_function, division
import time
from warnings import warn
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator, is_regressor, is_classifier, clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


_DEFAULT_IMPORTANCE = 'shap'

# Shap is nice to have
try:
    import shap
except ImportError:
    warn('SHAP not installed, install it if you want to use Shapley values as feature importance\n'
         'Reverting to permutation importance as default')
    _DEFAULT_IMPORTANCE = 'pimp'

# MPL is a must
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

# Progress bar to notify Users is a best practice
try:
    from tqdm import tqdm

    TQDM_INSTALLED = True
except ImportError:
    TQDM_INSTALLED = False

# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""


class BorutaPy(BaseEstimator, TransformerMixin):
    """
    This is a PR version of BorutaPy which itself is an
    improved Python implementation of the Boruta R package.
    For chronological dev, see https://github.com/scikit-learn-contrib/boruta_py/pull/77

    This PR vs BorutaPy:
    ------------------

    To summarize, this PR solves/enhances:
     - The categorical features (they are detected, encoded. The tree-based models are working
       better with integer encoding rather than with OHE, which leads to deep and unstable trees).
       If Catboost is used, then the cat.pred (if any) are set up
     - Work with Catboost sklearn API
     - Allow using sample_weight, for applications like Poisson regression or
       any requiring weights
     - 3 different feature importances: native, SHAP and permutation.
       Native being the least consistent
       (because of the imp. biased towards numerical and large cardinality categorical)
       but the fastest of the 3.
     - Visualization like in the R package

    BorutaPy vs Boruta R:
    ---------------------

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
        in each iteration based on the number of features under investigation.
        This way more trees are used when the training data has many features
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


    Summary
    -------
         - Loop over n_iter or until dec_reg == 0
         - add shadows
            o find features that are tentative
            o make sure that at least 5 columns are added
            o shuffle shadows
            o get feature importance
                * fit the estimator
                * extract feature importance (native, shap or permutation)
                * return feature importance
            o separate the importance of shadows and real

         - Calculate the maximum shadow importance and append to the previous run
         - Assign hits using the imp_sha_max of this run
            o find all the feat imp > imp_sha_max
            o tag them as hits
            o add +1 to the previous tag vector
         - Perform a test
            o select non rejected features yet
            o get a binomial p-values (nbr of times the feat has been tagged as important
            on the n_iter done so far) o reject or not according the (corrected) p-val


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
        vs shadow predictors. Note that the builtin tree importance (gini/impurity based
        importance) is biased towards numerical and large cardinality predictors, even
        if they are random. Shapley values and permutation imp. are robust w.r.t those predictors.
        Possible values: 'shap' (Shapley values),
        'pimp' (permutation importance) and 'native' (Gini/impurity)
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

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already

    early_stopping : bool, default = False
        Whether to use early stopping to terminate the selection process
        before reaching `max_iter` iterations if the algorithm cannot
        confirm a tentative feature for `n_iter_no_change` iterations.
        Will speed up the process at a cost of a possibility of a
        worse result.

    n_iter_no_change : int, default = 20
        Ignored if `early_stopping` is False. The maximum amount of
        iterations without confirming a tentative feature.

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
    from lightgbm import LGBMClassifier
    from boruta import BorutaPy

    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    y = y.ravel()

    # define lightGBM classifier, with utilising all cores and
    # sampling in proportion to y labels
    lgbc = LGBMClassifier(max_depth=5)


    # define Boruta feature selection method
    feat_selector = BorutaPy(lgbc, n_estimators='auto', verbose=2, random_state=1)

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

    def __init__(self,
                 estimator,
                 n_estimators=1000,
                 perc=90,
                 alpha=0.05,
                 importance=_DEFAULT_IMPORTANCE,
                 two_step=True,
                 max_iter=100,
                 random_state=None,
                 verbose=0,
                 early_stopping=False,
                 n_iter_no_change=20):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.importance = importance
        self.cat_name = None
        self.cat_idx = None
        # Catboost doesn't allow to change random seed after fitting
        self.is_tree_based = check_if_tree_based(self.estimator)
        self._is_lightgbm = 'lightgbm' in str(type(self.estimator))
        self.is_cat = 'catboost' in str(type(self.estimator))
        self.is_xgb = 'xgboost' in str(type(self.estimator))
        # plotting
        self.imp_real_hist = None
        self.sha_max = None
        self.col_names = None
        self.tag_df = None
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.__version__ = '0.4'

        # Hackish but avoid direct dependency to the SHAP package
        if self.importance == 'shap' and (not self.is_tree_based):
            raise TypeError("To compute SHAP feature importance, use a tree-based model")

    def fit(self, X, y, sample_weight=None):
        """
        Fits the Boruta feature selection with the provided estimator.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample
        """
        self.imp_real_hist = np.empty((0, X.shape[1]), float)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.col_names = X.columns.to_list()

        return self._fit(X, y, sample_weight=sample_weight)

    def transform(self, X, weak=False, return_df=False):
        """
        Reduces the input X to the features selected by Boruta.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        weak : boolean, default = False
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

    def fit_transform(self, X, y, sample_weight=None, weak=False, return_df=False):
        """
        Fits Boruta, then reduces the input X to the selected features.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample

        weak : boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.

        Returns
        -------

        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.

        Summary
        -------

         - Loop over n_iter or until dec_reg == 0
         - add shadows
            o find features that are tentative
            o make sure that at least 5 columns are added
            o shuffle shadows
            o get feature importance
                * fit the estimator
                * extract feature importance (native, shap or permutation)
                * return feature importance
            o separate the importance of shadows and real

         - Calculate the maximum shadow importance and append to the previous run
         - Assign hits using the imp_sha_max of this run
            o find all the feat imp > imp_sha_max
            o tag them as hits
            o add +1 to the previous tag vector
         - Perform a test
            o select non rejected features yet
            o get a binomial p-values (nbr of times the feat has been tagged as
            important on the n_iter done so far) o reject or not according the (corrected) p-val
        """

        self._fit(X, y, sample_weight=sample_weight)
        return self._transform(X, weak, return_df)

    def plot_importance(self, n_feat_per_inch=5):
        """
        Boxplot of the variable importance, ordered by magnitude
        The max shadow variable importance illustrated by the dashed line.
        Requires to apply the fit method first.

        Parameters
        ----------

        n_feat_per_inch : float
            the number of boxes per inch (vertical axis). Increace to make the chart smaller and denser,
            decrease this number for better readability

        Returns
        -------

        fig : matplotlib.figure.Figure
            the box plot of the Boruta feature importance and the statistical test results
            (accepted, tentative, rejected)


        """
        # plt.style.use('fivethirtyeight')

        if not MATPLOTLIB_INSTALLED:
            raise ImportError('You must install matplotlib and restart your session to plot importance.')

        my_colors_list = ['#000000', '#7F3C8D', '#11A579', '#3969AC',
                          '#F2B701', '#E73F74', '#80BA5A', '#E68310',
                          '#008695', '#CF1C90', '#F97B72']
        bckgnd_color = "#f5f5f5"
        params = {"axes.prop_cycle": plt.cycler(color=my_colors_list),
                  "axes.facecolor": bckgnd_color, "patch.edgecolor": bckgnd_color,
                  "figure.facecolor": bckgnd_color,
                  "axes.edgecolor": bckgnd_color, "savefig.edgecolor": bckgnd_color,
                  "savefig.facecolor": bckgnd_color, "grid.color": "#d2d2d2",
                  'lines.linewidth': 1.5}  # plt.cycler(color=my_colors_list)
        mpl.rcParams.update(params)

        if self.imp_real_hist is None:
            raise ValueError("Use the fit method first to compute the var.imp")

        color = {'boxes': 'gray', 'whiskers': 'gray', 'medians': '#000000', 'caps': 'gray'}
        vimp_df = pd.DataFrame(self.imp_real_hist, columns=self.col_names)
        vimp_df = vimp_df.reindex(vimp_df.mean().sort_values(ascending=True).index, axis=1)
        bp = vimp_df.boxplot(color=color,
                             boxprops=dict(linestyle='-', linewidth=1.5),
                             flierprops=dict(linestyle='-', linewidth=1.5),
                             medianprops=dict(linestyle='-', linewidth=1.5, color='#000000'),
                             whiskerprops=dict(linestyle='-', linewidth=1.5),
                             capprops=dict(linestyle='-', linewidth=1.5),
                             showfliers=False, grid=True, rot=0, vert=False, patch_artist=True,
                             figsize=(16, vimp_df.shape[1] / n_feat_per_inch), fontsize=9
                             )
        blue_color = "#2590fa"
        yellow_color = "#f0be00"
        n_strong = sum(self.support_)
        n_weak = sum(self.support_weak_)
        n_discarded = len(self.col_names) - n_weak - n_strong
        box_face_col = [blue_color] * n_strong + [yellow_color] * n_weak + ['gray'] * n_discarded
        for c in range(len(box_face_col)):
            bp.findobj(mpl.patches.Patch)[len(self.support_) - c - 1].set_facecolor(box_face_col[c])
            bp.findobj(mpl.patches.Patch)[len(self.support_) - c - 1].set_color(box_face_col[c])

        xrange = vimp_df.max(skipna=True).max(skipna=True) - vimp_df.min(skipna=True).min(skipna=True)
        bp.set_xlim(left=vimp_df.min(skipna=True).min(skipna=True) - 0.10 * xrange)

        custom_lines = [Line2D([0], [0], color=blue_color, lw=5),
                        Line2D([0], [0], color=yellow_color, lw=5),
                        Line2D([0], [0], color="gray", lw=5),
                        Line2D([0], [0], linestyle='--', color="gray", lw=2)]
        bp.legend(custom_lines, ['confirmed', 'tentative', 'rejected', 'sha. max'], loc="lower right")
        plt.axvline(x=self.sha_max, linestyle='--', color='gray')
        fig = bp.get_figure()
        plt.title('Boruta importance and selected predictors')
        # fig.set_size_inches((10, 1.5 * np.rint(max(vimp_df.shape) / 10)))
        # plt.tight_layout()
        # plt.show()
        return fig

    @staticmethod
    def _validate_pandas_input(arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    def _fit(self, X_raw, y, sample_weight=None):
        """
        Private method.
        Chaining:
         - Loop over n_iter or until dec_reg == 0
         - add shadows
            o find features that are tentative
            o make sure that at least 5 columns are added
            o shuffle shadows
            o get feature importance
                * fit the estimator
                * extract feature importance (native, shap or permutation)
                * return feature importance
            o separate the importance of shadows and real

         - Calculate the maximum shadow importance and append to the previous run
         - Assign hits using the imp_sha_max of this run
            o find all the feat imp > imp_sha_max
            o tag them as hits
            o add +1 to the previous tag vector
         - Perform a test
            o select non rejected features yet
            o get a binomial p-values (nbr of times the feat has been tagged as
            important on the n_iter done so far) o reject or not according the (corrected) p-val

        Parameters
        ----------

        X_raw : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample

        Returns
        -------

         self : object
            Nothing but attributes
        """
        # self.is_cat = is_catboost(self.estimator)

        start_time = time.time()
        # basic cat features encoding
        # First, let's store "object" columns as categorical columns
        # obj_feat = X_raw.dtypes.loc[X_raw.dtypes == 'object'].index.tolist()
        obj_feat = list(set(list(X_raw.columns)) - set(list(X_raw.select_dtypes(include=[np.number]))))
        X = X_raw
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        # w = w.fillna(0)

        self.cat_name = obj_feat
        # self.cat_idx = cat_idx

        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X)
        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                sample_weight = self._validate_pandas_input(sample_weight)

        self.random_state = check_random_state(self.random_state)

        early_stopping = False
        if self.early_stopping:
            if self.n_iter_no_change >= self.max_iter:
                if self.verbose > 0:
                    print(
                        f"n_iter_no_change is bigger or equal to max_iter"
                        f"({self.n_iter_no_change} >= {self.max_iter}), "
                        f"early stopping will not be used."
                    )
            else:
                early_stopping = True

        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # early stopping vars
        _same_iters = 1
        _last_dec_reg = None
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
        if TQDM_INSTALLED:
            pbar = tqdm(total=self.max_iter, desc="Boruta iteration")
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            # be sure to use an non-fitted estimator
            self.estimator = clone(self.estimator)
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            # Catboost doesn't allow to change random seed after fitting
            if self.is_cat is False:
                if self._is_lightgbm:
                    self.estimator.set_params(random_state=self.random_state.randint(0, 10000))
                else:
                    self.estimator.set_params(random_state=self.random_state)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, sample_weight, dec_reg)

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
            # if self.verbose > 0 and _iter < self.max_iter:
            #     self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1
                if TQDM_INSTALLED:
                    pbar.update(1)

            # early stopping
            if early_stopping:
                if _last_dec_reg is not None and (_last_dec_reg == dec_reg).all():
                    _same_iters += 1
                    if self.verbose > 0:
                        print(
                            f"Early stopping: {_same_iters} out "
                            f"of {self.n_iter_no_change}"
                        )
                else:
                    _same_iters = 1
                    _last_dec_reg = dec_reg.copy()
                if _same_iters > self.n_iter_no_change:
                    break

        if TQDM_INSTALLED:
            pbar.close()
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

        if isinstance(X_raw, np.ndarray):
            X_raw = pd.DataFrame(X_raw)

        if isinstance(X_raw, pd.DataFrame):
            self.support_names_ = [X_raw.columns[i] for i, x in enumerate(self.support_) if x]
            self.tag_df = pd.DataFrame({'predictor': list(X_raw.columns)})
            self.tag_df['Boruta'] = 1
            self.tag_df['Boruta'] = np.where(self.tag_df['predictor'].isin(list(self.support_names_)), 1, 0)

        if isinstance(X_raw, pd.DataFrame):
            self.support_weak_names_ = [X_raw.columns[i] for i, x in enumerate(self.support_weak_) if x]
            self.tag_df['Boruta_weak_incl'] = np.where(self.tag_df['predictor'].isin(
                list(self.support_names_ + self.support_weak_names_)
            ), 1, 0)

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
        print("All relevant predictors selected in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                                                                  int(minutes),
                                                                                  seconds))
        return self

    def _transform(self, X, weak=False, return_df=False):
        """
        Private method

        transform the predictor matrix by dropping the rejected and
        (optional) the undecided predictors

        Parameters
        ----------
        X : pd.DataFrame
            predictor matrix
        weak : bool
            whether to drop or not the undecided predictors
        return_df : bool
            return a pandas dataframe or not

        Returns
        -------

         X : np.array or pd.DataFrame
            the transformed predictors matrix
        """
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
        depth = None
        try:
            depth = self.estimator.get_params()['max_depth']
        except KeyError:
            warn(
                "The estimator does not have a max_depth property, as a result "
                " the number of trees to use cannot be estimated automatically."
            )
        if depth is None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        # n_feat * 2 because the training matrix is extended with n shadow features
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _get_shuffle(self, seq):
        self.random_state.shuffle(seq)
        return seq

    def _add_shadows_get_imps(self, X, y, sample_weight, dec_reg):
        """
        Add a shuffled copy of the columns (shadows) and get the feature
        importance of the augmented data set

        Parameters
        ----------

        X : pd.DataFrame of shape [n_samples, n_features]
            predictor matrix
        y : pd.series of shape [n_samples]
            target
        sample_weight : array-like, shape = [n_samples], default=None
            Individual weights for each sample
        dec_reg : array
            holds the decision about each feature 1, 0, -1 (accepted, undecided, rejected)

        Returns
        -------

         imp_real : array
            feature importance of the real predictors
         imp_sha : array
            feature importance of the shadow predictors
        """
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = np.copy(x_cur)
        # make sure there's at least 5 columns in the shadow matrix for
        while x_sha.shape[1] < 5:
            x_sha = np.hstack((x_sha, x_sha))
        # shuffle xSha
        x_sha = np.apply_along_axis(self._get_shuffle, 0, x_sha)
        # get importance of the merged matrix
        if self.importance == 'shap':
            imp = _get_shap_imp(self.estimator,
                                np.hstack((x_cur, x_sha)),
                                y,
                                sample_weight)
        elif self.importance == 'pimp':
            imp = _get_perm_imp(self.estimator, np.hstack((x_cur, x_sha)), y, sample_weight)
        else:
            imp = _get_imp(self.estimator, np.hstack((x_cur, x_sha)), y, sample_weight)

        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]
        return imp_real, imp_sha

    @staticmethod
    def _assign_hits(hit_reg, cur_imp, imp_sha_max):
        """
        count how many times a given feature was more important than
        the best of the shadow features

        Parameters
        ----------

        hit_reg : array
            count how many times a given feature was more important than the
            best of the shadow features
        cur_imp : array
            current importance
        imp_sha_max : array
            importance of the best shadow predictor

        Returns
        -------

        hit_reg : array
            count how many times a given feature was more important than the
            best of the shadow features
        """
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp[0]
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        """
        Perform the rest if the feature should be tagged as relevant (confirmed), not relevant (rejected)
        or undecided. The test is performed by considering the binomial tentatives over several attempts.
        I.e. count how many times a given feature was more important than the best of the shadow features
        and test if the associated probability to the z-score is below, between or above the rejection or
        acceptance threshold.

        Parameters
        ----------

        dec_reg : array
            holds the decision about each feature 1, 0, -1 (accepted, undecided, rejected)
        hit_reg : array
            counts how many times a given feature was more important than the best of the shadow features
        _iter: int

        Returns
        -------

        dec_reg = np.array
            the tag array
        """
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

    @staticmethod
    def _fdrcorrection(pvals, alpha=0.05):
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

    @staticmethod
    def _nanrankdata(X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Private method
        Check hyperparameters as well as X and y before proceeding with fit.

        Returns
        -------

        X : pd.DataFrame
            predictor matrix
        y : pd.series
            target
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y, dtype=None, force_all_finite=False)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _print_results(self, dec_reg, _iter, flag):
        """
        Private method
        printing the result

        Params
        ------

        dec_reg : np.array
            if the feature as been tagged as relevant (confirmed),
            not relevant (rejected) or undecided
        _iter : int
            the iteration number
        flag : int
            is still in the feature selection process or not

        Returns
        -------

         output : str
            the output to be printed out
        """
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
            n_rejected = np.sum(~(self.support_ | self.support_weak_))
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            if self.importance in ['shap', 'pimp']:
                vimp = str(self.importance)
            else:
                vimp = 'native'
            output = "\n\nBoruta finished running using " + vimp + " var. imp.\n\n" + result
        print(output)


def _split_fit_estimator(estimator, X, y, sample_weight=None, cat_feature=None):
    """
    Private function
    split the train, test and fit the model

    Parameters
    ----------

    estimator: sklearn estimator
        the scikit-learn estimator
    X: pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    y: pd.series of shape [n_samples]
        target
    sample_weight: array-like, shape = [n_samples], default=None
        Individual weights for each sample
    cat_feature: list of int or None
        the list of integers, cols loc, of the categrocial predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.


    Returns
    -------

     model: sklearn estimator
        fitted model
     X_tt: array [n_samples, n_features]
        the test split, predictors
     y_tt: array [n_samples]
        the test split, target
    """

    if cat_feature is None:
        # detect, store and encode categorical predictors
        X = pd.DataFrame(X)
        obj_feat = list(set(list(X.columns)) - set(list(X.select_dtypes(include=[np.number]))))
        if obj_feat:
            X[obj_feat] = X[obj_feat].astype('str').astype('category')
            for col in obj_feat:
                X[col] = X[col].astype('category').cat.codes
        else:
            obj_feat = None
    else:
        obj_feat = cat_feature

    if sample_weight is not None:
        w = sample_weight
        if is_regressor(estimator):
            X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, random_state=42)
        else:
            X_tr, X_tt, y_tr, y_tt, w_tr, w_tt = train_test_split(X, y, w, stratify=y, random_state=42)
    else:
        if is_regressor(estimator):
            X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, random_state=42)
        else:
            X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, stratify=y, random_state=42)
        w_tr, w_tt = None, None

    X_tr = pd.DataFrame(X_tr)
    X_tt = pd.DataFrame(X_tt)

    if check_if_tree_based(estimator):
        try:
            # handle cat features if supported by the fit method
            if is_catboost(estimator) or ('cat_feature' in estimator.fit.__code__.co_varnames):
                model = estimator.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=obj_feat, verbose=0)
            else:
                model = estimator.fit(X_tr, y_tr, sample_weight=w_tr, verbose=0)

        except Exception as e:
            raise ValueError('Please check your X and y variable. The provided '
                             'estimator cannot be fitted to your data.\n' + str(e))
    else:
        raise ValueError('Not a tree based model')

    return model, X_tt, y_tt, w_tt


def _get_shap_imp(estimator, X, y, sample_weight=None, cat_feature=None):
    """
    Private function
    Get the SHAP feature importance

    Parameters
    ----------

    estimator: sklearn estimator
        the scikit-learn estimator
    X: pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    y: pd.series of shape [n_samples]
        target
    sample_weight: array-like, shape = [n_samples], default=None
        Individual weights for each sample
    cat_feature: list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.

    Returns
    -------

     shap_imp: array
        the SHAP importance array
    """

    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    model, X_tt, y_tt, w_tt = _split_fit_estimator(estimator, X, y,
                                                   sample_weight=sample_weight,
                                                   cat_feature=cat_feature)

    # Faster and safer to use the builtin lightGBM method
    # Note the xgboost and catboost have builtin shap as well
    # but it requires to use DMatrix or Pool respectively
    # for other tree-based models, no builtin SHAP
    if is_lightgbm(estimator):
        shap_matrix = model.predict(X_tt, pred_contrib=True)
        # the dim changed in lightGBM 3
        # X_SHAP_values array-like of shape = [n_samples, n_features + 1] or 
        # shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
        if is_classifier(estimator):
            # remove every (n_features+1)th element, python indexes from zero
            n_features = X_tt.shape[1]
            shap_matrix = np.delete(shap_matrix, list(range(n_features, shap_matrix.shape[1], n_features+1)), axis=1)
            shap_imp = np.mean(np.abs(shap_matrix), axis=0)
        else:
            shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
        
    else:
        # build the explainer
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_tt)
        # flatten to 2D if classification and lightgbm
        if is_classifier(estimator):
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

    return shap_imp


def _get_perm_imp(estimator, X, y, sample_weight, cat_feature=None):
    """
    Private function
    Get the permutation feature importance

    Parameters
    ----------

    estimator: sklearn estimator
        the scikit-learn estimator
    X: pd.DataFrame of shape [n_samples, n_features]
        predictor matrix
    y: pd.series of shape [n_samples]
        target
    sample_weight: array-like, shape = [n_samples], default=None
        Individual weights for each sample
    cat_feature: list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.

    Returns
    -------

     imp: array
        the permutation importance array
    """
    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    model, X_tt, y_tt, w_tt = _split_fit_estimator(estimator, X, y,
                                                   sample_weight=sample_weight,
                                                   cat_feature=cat_feature)
    perm_imp = permutation_importance(model, X_tt, y_tt, n_repeats=5, random_state=42, n_jobs=-1)
    imp = perm_imp.importances_mean.ravel()
    return imp


def _get_imp(estimator, X, y, sample_weight=None, cat_feature=None):
    """
    Get the native feature importance (impurity based for instance)
    This is know to return biased and uninformative results.
    e.g.
    https://scikit-learn.org/stable/auto_examples/inspection/
    plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

    or
    https://explained.ai/rf-importance/

    Parameters
    ----------

    X : array-like, shape = [n_samples, n_features]
        The training input samples.
    y : array-like, shape = [n_samples]
        The target values.
    sample_weight : array-like, shape = [n_samples], default=None
        Individual weights for each sample
    cat_feature: list of int or None
        the list of integers, cols loc, of the categorical predictors. Avoids to detect and encode
        each iteration if the exact same columns are passed to the selection methods.

    Returns
    -------

    imp: np.array
        the importance array
    """
    # be sure to use an non-fitted estimator
    estimator = clone(estimator)

    try:
        # handle categoricals
        X = pd.DataFrame(X)
        if cat_feature is None:
            obj_feat = list(set(list(X.columns)) - set(list(X.select_dtypes(include=[np.number]))))
            if obj_feat:
                X[obj_feat] = X[obj_feat].astype('str').astype('category')
                for col in obj_feat:
                    X[col] = X[col].cat.codes
            else:
                obj_feat = None
        else:
            obj_feat = cat_feature

        # handle catboost and cat features
        if is_catboost(estimator) or ('cat_feature' in estimator.fit.__code__.co_varnames):
            X = pd.DataFrame(X)
            estimator.fit(X, y, sample_weight=sample_weight, cat_features=obj_feat, verbose=0)
        else:
            estimator.fit(X, y, sample_weight=sample_weight, verbose=0)

    except Exception as e:
        raise ValueError('Please check your X and y variable. The provided '
                         'estimator cannot be fitted to your data.\n' + str(e))
    try:
        imp = estimator.feature_importances_
    except Exception:
        raise ValueError('Only methods with feature_importance_ attribute '
                         'are currently supported in BorutaPy.')
    return imp


# for better looking conditions
def is_lightgbm(estimator):
    """helper is lightGBM model (better looking condition in private func)"""
    is_lgb = 'lightgbm' in str(type(estimator))
    return is_lgb


def is_catboost(estimator):
    """helper is catboost model (better looking condition in private func)"""
    is_cat = 'catboost' in str(type(estimator))
    return is_cat


def is_xgboost(estimator):
    """helper is xgboost model (better looking condition in private func)"""
    is_xgb = 'xgboost' in str(type(estimator))
    return is_xgb


def check_if_tree_based(model):
    """helper is tree-based model (better looking condition in private func)"""
    tree_based_models = ['lightgbm', 'xgboost', 'catboost', '_forest', 'boosting']
    condition = any(i in str(type(model)).lower() for i in tree_based_models)
    return condition
