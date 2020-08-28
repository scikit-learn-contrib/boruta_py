%matplotlib inline
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.inspection import permutation_importance
import catboost
from boruta import BorutaPy as bp
from sklearn.datasets import load_boston, load_diabetes
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import gc
import shap
# lightgbm and catboost
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sys import getsizeof, path
plt.style.use('fivethirtyeight')
rng = np.random.RandomState(seed=42)



# Convert the cat. pred. for boosting
def cat_var(df, col_excl=None, return_cat=True):
    """Identify categorical features.

        Parameters
        ----------
        df: original df after missing operations

        Returns
        -------
        cat_var_df: summary df with col index and col name for all categorical vars
        :param return_cat: Boolean, return encoded cols as type 'category'
        :param df: pd.DF, the encoded data-frame
        :param col_excl: list, colums not to be encoded
        """

    if col_excl is None:
        non_num_cols = list(set(list(df.columns)) - set(list(df.select_dtypes(include=[np.number]))))
    else:
        non_num_cols = list(
            set(list(df.columns)) - set(list(df.select_dtypes(include=[np.number]))) - set(col_excl))

    # cat_var_index = [i for i, x in enumerate(df[col_names].dtypes.tolist()) if isinstance(x, pd.CategoricalDtype) or x == 'object']
    # cat_var_name = [x for i, x in enumerate(col_names) if i in cat_var_index]

    cat_var_index = [df.columns.get_loc(c) for c in non_num_cols if c in df]

    cat_var_df = pd.DataFrame({'cat_ind': cat_var_index,
                               'cat_name': non_num_cols})

    cols_need_mapped = cat_var_df.cat_name.to_list()
    inv_mapper = {col: dict(enumerate(df[col].astype('category').cat.categories)) for col in df[cols_need_mapped]}
    mapper = {col: {v: k for k, v in inv_mapper[col].items()} for col in df[cols_need_mapped]}

    for c in cols_need_mapped:
        df.loc[:, c] = df.loc[:, c].map(mapper[c]).fillna(0).astype(int)

    if return_cat:
        df[non_num_cols] = df[non_num_cols].astype('category')
    return df, cat_var_df, inv_mapper


def get_titanic_data():
    # Fetch Titanic data and add random cat and numbers
    # Example taken from https://scikit-learn.org/stable/auto_examples/inspection/
    # plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    rng = np.random.RandomState(seed=42)
    X['random_cat'] = rng.randint(3, size=X.shape[0])
    X['random_cat'] = X['random_cat'].astype('str')
    X['random_num'] = rng.randn(X.shape[0])

    categorical_columns = ['pclass', 'sex', 'embarked', 'random_cat']
    numerical_columns = ['age', 'sibsp', 'parch', 'fare', 'random_num']
    X = X[categorical_columns + numerical_columns]
    # Impute
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing'))])
    numerical_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    preprocessing = ColumnTransformer([('cat', categorical_pipe, categorical_columns), ('num', numerical_pipe, numerical_columns)])
    X_trans = preprocessing.fit_transform(X)
    X = pd.DataFrame(X_trans, columns = X.columns)
    # encode
    X, cat_var_df, inv_mapper = cat_var(X)
    # sample weight is just a dummy random vector for testing purpose
    sample_weight = np.random.uniform(0,1, len(y))
    return X, y, sample_weight

def get_boston_data():
    boston = load_boston()
    X = pd.DataFrame(boston.data)
    X['random_num1'] = rng.randn(X.shape[0])
    X['random_num2'] = np.random.poisson(1, X.shape[0])
    y = pd.Series(boston.target)
    return X, y

# Testing the changes with rnd cat. and num. predictors added to the set of genuine predictors
def testing_estimators(varimp, models, X, y, sample_weight=None):
    for model in models:
        print('='*20 +' testing: {mod:>55} for var.imp: {vimp:<15} '.format(mod=str(model), vimp=varimp)+'='*20 )
        feat_selector = noglmgroot.BorutaPy(model, n_estimators = 100, verbose= 1, max_iter= 10, random_state=42, weight=None, importance=varimp)
        feat_selector.fit(X, y, sample_weight)
        print(feat_selector.support_names_)
        feat_selector.plot_importance(n_feat_per_inch=3)
        gc.enable()
        del(feat_selector, model)
        gc.collect()

def testing_clf_all_varimp(X, y, sample_weight=None):
    for varimp in ['shap', 'pimp', 'native']:
        models = [RandomForestClassifier(n_jobs= 4, oob_score= True), catboost.CatBoostClassifier(random_state=42, verbose=0), XGBClassifier(random_state=42, verbose=0), LGBMClassifier(random_state=42, verbose=0)]
        testing_estimators(varimp=varimp, models=models, X=X, y=y, sample_weight=sample_weight)
        gc.enable()
        del models
        gc.collect()

def testing_regr_all_varimp(X, y):
    for varimp in ['shap', 'pimp', 'native']:
        models = [catboost.CatBoostRegressor(random_state=42, verbose=0), XGBRegressor(random_state=42, verbose=0), LGBMRegressor(random_state=42, verbose=0)]
        testing_estimators(varimp=varimp, models=models, X=X, y=y)
        gc.enable()
        del(models)
        gc.collect()

print('='*20 + ' Benchmarking using sklearn permutation importance ' + '='*20 )
X, y, sample_weight = get_titanic_data()
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weight, stratify=y, random_state=42)
# lightgbm faster and better than RF
lgb_model = LGBMClassifier(n_jobs= 4)
lgb_model.fit(X_train, y_train, sample_weight=w_train)
result = permutation_importance(lgb_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
# Plot
fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()

if __name__ == '__main__':
    # classification and cat. pred, sample weight is just a dummy random vector for testing purpose
    X, y, sample_weight = get_titanic_data() #get_titanic_data()
    testing_clf_all_varimp(X=X, y=y, sample_weight=sample_weight)

    # regression
    X, y = get_boston_data()
    testing_regr_all_varimp(X=X, y=y)