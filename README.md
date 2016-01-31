# README #

This project hosts Python implementations of the [Boruta all-relevant feature selection method](https://m2.icm.edu.pl/boruta/).

### How do I get set up? ###

* You'll need numpy, scipy, bottleneck, statsmodels and scikit-learn to run this.
* Just download, import into your project and do as you would with any other 
sklearn method: fit(X, y), transform(X) and fit_transform(X, y) methods are 
implemented.

* * 

### Description ###

Python implementations of the Boruta R package.

This implementation tries to mimic the scikit-learn interface, so use fit,
transform or fit_transform, to run the feautre selection.

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

* * *

### Versions ###

BorutaPy
--------

It is the original R package recoded in Python with a few added extra features.
Some improvements include:  

* Faster run times, thanks to scikit-learn

* Scikit-learn like interface

* Compatible with any ensemble method from scikit-learn

* Automatic n_estimator selection

* Ranking of features
    
For more details, please check the top of the docstring.

We highly recommend using pruned trees with a depth between 3-7.

BorutaPy2
---------

After playing around a lot with the original code I identified a few areas 
where the core algorithm could be improved. I basically ran lots of 
benchmarking tests on simulated datasets (scikit-learn's amazing 
make_classification was used for generating these). 

__Percentile as threshold__  
The original method uses the maximum of the shadow features as a threshold in
deciding which real feature is doing better than the shadow ones. This could be
overly harsh. 

To control this, in the 2nd version the perc parameter sets the
percentile of the shadow features' importances, the algorithm uses as the 
threshold. The default of 99 is close to taking the maximum, but as it's a
percentile, it changes with the size of the dataset. With several thousands of 
features it isn't as stringent as with a few dozens at the end of a Boruta run.


__Two step correction for multiple testing__  
The correction for multiple testing was improved by making it a two step 
process, rather than a harsh one step Bonferroni correction.

We need to correct firstly because in each iteration we test a number of 
features against the null hypothesis (does a feature perform better than
expected by random). For this the Bonferroni correction is used in the original 
code which is known to be too stringent in such scenarios, and also the 
original code corrects for n features, even if we are in the 50th iteration 
where we only have k<<n features left. For this reason the first step of 
 correction is the widely used Benjamini Hochberg FDR. 
 
Following that however we also need to account for the fact that we have been
testing the same features over and over again in each iteration with the
same test. For this scenario the Bonferroni is perfect, so it is applied by
deviding the p-value threshold with the current iteration index.
 
We highly recommend using pruned trees with a depth between 3-7.

* * *

### Parameters, attributes, examples, reference ###

Parameters
----------

__estimator__ : object
   > A supervised learning estimator, with a 'fit' method that returns the
   > feature_importances_ attribute. Important features must correspond to
   > high absolute values in the feature_importances_.

__n_estimators__ : int or string, default = 1000
   > If int sets the number of estimators in the chosen ensemble method.
   > If 'auto' this is determined automatically based on the size of the
   > dataset. The other parameters of the used estimators need to be set
   > with initialisation.

__multi_corr_method__ : string, default = 'bonferroni' - only in BorutaPy
>Method for correcting for multiple testing during the feature selection process. statsmodels' multiple test is used, so one of the following:

>* 'bonferroni' : one-step correction
>* 'sidak' : one-step correction
>* 'holm-sidak' : step down method using Sidak adjustments
>* 'holm' : step-down method using Bonferroni adjustments
>* 'simes-hochberg' : step-up method  (independent)
>* 'hommel' : closed method based on Simes tests (non-negative)
>* 'fdr_bh' : Benjamini/Hochberg  (non-negative)
>* 'fdr_by' : Benjamini/Yekutieli (negative)
>* 'fdr_tsbh' : two stage fdr correction (non-negative)
>* 'fdr_tsbky' : two stage fdr correction (non-negative)

__perc__ : int, default = 99 - only in BorutaPy2
   > Instead of the max we use the percentile defined by the user, to pick
   > our threshold for comparison between shadow and real features. The max
   > tend to be too stringent. This provides a finer control over this. The
   > lower perc is the more false positives will be picked as relevant but 
   > also the less relevant features will be left out. The usual trade-off.


__multi_alpha__ : float, default = 0.05
   > Level at which the corrected p-values will get rejected in both correction
   steps.

__max_iter__ : int, default = 100
   > The number of maximum iterations to perform.

__verbose__ : int, default=0
   > Controls verbosity of output.

* * *

Attributes
----------

**n_features_** : int
   > The number of selected features.

**support_** : array of shape [n_features]
   > The mask of selected features - only confirmed ones are True.

**support_weak_** : array of shape [n_features]
  >  The mask of selected tentative features, which haven't gained enough
  >  support during the max_iter number of iterations..

**ranking_** : array of shape [n_features]
  >  The feature ranking, such that ``ranking_[i]`` corresponds to the
  >  ranking position of the i-th feature. Selected (i.e., estimated
  >  best) features are assigned rank 1 and tentative features are assigned
  >  rank 2.

* * *

Examples
--------

    import pandas
    from sklearn.ensemble import RandomForestClassifier
    from boruta_py import boruta_py

    # load X and y
    X = pd.read_csv('my_X_table.csv', index_col=0).values
    y = pd.read_csv('my_y_vector.csv', index_col=0).values

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)

    # define Boruta feature selection method
    feat_selector = boruta_py.BorutaPy(rf, n_estimators='auto', verbose=2)

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

* * *

References
----------

[1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
    Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
