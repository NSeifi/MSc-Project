"""
Author: Nasrin Seifi
Input: "company_preprocessed_csv" ticker datasets
Functionality: Runs the data sets through different feature selectors available in sklearn library and returns the best
    set, based on the performance of the selected feature set in a baseline decision tree.
Output: TBD

Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        dataset-creation/create-nasdaq100-dataset.py
"""
from sklearn.feature_selection import *
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
import pandas as pd
from tqdm import tqdm
import warnings
from glob import glob

warnings.filterwarnings('ignore')


def SelectFdrF(X_train, y_train, X_test):
    transformer = SelectFdr(f_classif, alpha=0.01)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectFprF2(X_train, y_train, X_test):
    transformer = SelectFpr(f_regression, alpha=0.01)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectFweF2(X_train, y_train, X_test):
    transformer = SelectFwe(f_regression, alpha=0.01)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectKBestF(X_train, y_train, X_test):
    transformer = SelectKBest(f_classif, k=12)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectKBestF2(X_train, y_train, X_test):
    transformer = SelectKBest(f_regression, k=12)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectKBestF3(X_train, y_train, X_test):
    transformer = SelectKBest(r_regression, k=12)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectKBestF4(X_train, y_train, X_test):
    transformer = SelectKBest(mutual_info_regression, k=12)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectFromModelF2(X_train, y_train, X_test):
    transformer = SelectFromModel(estimator=Ridge(alpha=1, normalize=True))
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectFromModelF5(X_train, y_train, X_test):
    transformer = SelectFromModel(estimator=RandomForestRegressor(n_estimators=500, random_state=42))
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SelectFromModelF6(X_train, y_train, X_test):
    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_train, y_train)
    importance = np.abs(ridge.coef_)
    threshold = np.sort(importance)[-3] + 0.01
    transformer = SelectFromModel(ridge, threshold=threshold)
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SequentialFeatureSelectorF(X_train, y_train, X_test):
    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_train, y_train)
    transformer = SequentialFeatureSelector(ridge, n_features_to_select=10, direction="forward")
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def SequentialFeatureSelectorF2(X_train, y_train, X_test):
    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_train, y_train)
    transformer = SequentialFeatureSelector(ridge, n_features_to_select=10, direction="backward")
    transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def RFEF(X_train, y_train, X_test):
    estimator = SVR(kernel="linear")
    transformer = RFE(estimator, n_features_to_select=10, step=1)
    transformer = transformer.fit(X_train, y_train)
    mask = transformer.support_
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def VarianceThresholdF(X_train, y_train, X_test):
    transformer = VarianceThreshold(100)
    transformer = transformer.fit(X_train, y_train)
    mask = transformer.get_support()
    x_hat_train = transformer.transform(X_train)
    x_hat_test = transformer.transform(X_test)
    return [x for x, m in zip(X_train.columns, mask) if m], x_hat_train, x_hat_test, y_train


def avg_eucleadan_distance(y_p, y_test):
    assert len(y_p) == len(y_test)
    return sum([abs(a - p) / a for p, a in zip(y_p, y_test)]) / len(y_p)


def train_decision_tree_regressor(features, x_hat_train, x_hat_test, y_train, y_test, function_name):
    clf = tree.DecisionTreeRegressor(max_depth=5)
    if x_hat_train.shape[1] == 0:
        return features, 1.0, function_name, "decision_tree_regressor"
    clf = clf.fit(x_hat_train, y_train)
    y_p = clf.predict(x_hat_test)
    error = avg_eucleadan_distance(y_p, y_test)
    return features, error, function_name, "decision_tree_regressor"


def train_extra_tree_regressor(features, x_hat_train, x_hat_test, y_train, y_test, function_name):
    clf = tree.ExtraTreeRegressor()
    if x_hat_train.shape[1] == 0:
        return features, 1.0, function_name, "decision_tree_regressor"
    clf = clf.fit(x_hat_train, y_train)
    y_p = clf.predict(x_hat_test)
    error = avg_eucleadan_distance(y_p, y_test)
    return features, error, function_name, "extra_tree_regressor"


def train_gradient_boosting_regressor(features, x_hat_train, x_hat_test, y_train, y_test, function_name):
    clf = GradientBoostingRegressor(n_estimators=200, max_depth=10)
    if x_hat_train.shape[1] == 0:
        return features, 1.0, function_name, "decision_tree_regressor"
    clf = clf.fit(x_hat_train, y_train)
    y_p = clf.predict(x_hat_test)
    error = avg_eucleadan_distance(y_p, y_test)
    return features, error, function_name, "gradient_boosting_regressor"


def extract_features(ticker, verbose=False):
    if not ticker.startswith("company_preprocessed_csv"):
        file_name = f"company_preprocessed_csv/{ticker}.csv"
    else:
        assert ticker.endswith(".csv")
        file_name = ticker

    dtype = {'SHRCD': float, 'EXCHCD': float, 'SICCD': float, 'ISSUNO': float, 'HEXCD': float, 'HSICCD': float,
             'DLAMT': float, 'DLSTCD': float, 'SHRFLG': float, 'HSICMG': float, 'HSICIG': float, 'DISTCD': float,
             'DIVAMT': float, 'FACPR': float, 'FACSHR': float, 'ACPERM': float, 'ACCOMP': float, 'NWPERM': float,
             'DLRETX': float, 'DLPRC': float, 'DLRET': float, 'TRTSCD': float, 'NMSIND': float, 'MMCNT': float,
             'NSDINX': float, 'BIDLO': float, 'ASKHI': float, 'PRC': float, 'VOL': float, 'RET': float, 'BID': float,
             'ASK': float, 'SHROUT': float, 'CFACPR': float, 'CFACSHR': float, 'OPENPRC': float, 'NUMTRD': float,
             'RETX': float, 'vwretd': float, 'vwretx': float, 'ewretd': float, 'ewretx': float, 'sprtrn': float,
             'Mkt-RF': float, 'SMB': float, 'HML': float, 'RF': float, 'IVOL': float}
    drop_features = ['Unnamed: 0', 'CUSIP', 'PERMNO', 'date', 'NCUSIP', 'TICKER', 'COMNAM', 'PERMCO', 'NAMEENDT',
                     'SHRCLS', 'TSYMBOL', 'NAICS', 'PRIMEXCH', 'TRDSTAT', 'SECSTAT', 'NEXTDT', 'DLPDT', 'DCLRDT',
                     'PAYDT', 'RCRDDT', 'year_month', 'year', 'month']

    data = pd.read_csv(file_name)
    data = data[data['IVOL'].notna()]  # we don't want the records that do not have a valid IVOL as training data
    data = data.fillna(0)
    for key in dtype:
        data.loc[pd.to_numeric(data[key], errors='coerce').isnull(), key] = 0

    for x in drop_features:
        data = data.drop(columns=[x])

    y = data["IVOL"].astype(float)

    X = data.drop(columns=["IVOL"]).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

    functions = [SelectFdrF, SelectFprF2, SelectFweF2, SelectKBestF, SelectKBestF2, SelectKBestF3, SelectKBestF4,
                 SelectFromModelF2, SelectFromModelF5, SelectFromModelF6, SequentialFeatureSelectorF,
                 SequentialFeatureSelectorF2, RFEF, VarianceThresholdF]

    models = [train_decision_tree_regressor, train_extra_tree_regressor, train_gradient_boosting_regressor]

    res = sorted([trainer(*func(X_train, y_train, X_test), y_test, func.__name__)
                  for trainer in models for func in functions], key=lambda e: e[1])
    if verbose:
        print("res \n" + ",".join([f"{x[0]}, {x[1]}, {x[2]}, {x[3]}" for x in res]))
    bestperf = res[0]
    if verbose:
        print(bestperf)

    return bestperf


if __name__ == '__main__':
    with open("best_feature_set.csv", "w", encoding="utf-8") as bfs_file:
        bfs_file.write("Ticker, Features, BestErrorRate, FeatureSelector, BaselineModel\n")
        _itr = tqdm(glob("company_preprocessed_csv/*"))
        for t in _itr:
            _itr.set_description(f"Processing {t}")
            _ticker = t.split("/")[-1].split(".")[0]
            features_, best_error_rate, feature_selector_name, baseline_model_name = extract_features(t)
            bfs_file.write(f"{_ticker}, {'_'.join(features_)}, {best_error_rate}, {feature_selector_name},"
                           f" {baseline_model_name}\n")
