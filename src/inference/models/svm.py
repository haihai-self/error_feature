from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from evaluate import classify, regression


def classifySVM(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    param = {'C': [x / 10 for x in range(1, 21, 2)],
             'kernel': ['poly', 'rbf', 'sigmoid']}
    gsearch = GridSearchCV(estimator=svm.SVC(C=10, kernel='poly')
                           , param_grid=param, cv=5, n_jobs=-1, scoring=classify.score())
    gsearch.fit(x.values, y)
    C = gsearch.best_params_['C']
    kernel = gsearch.best_params_['kernel']
    model = svm.SVC(C=C, kernel=kernel)
    model.fit(x, y)
    print(gsearch.best_params_)
    return model


def classifySVMPoly(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    model_poly = svm.SVC(C=10, kernel='poly', degree=5)
    model_poly.fit(x, y)

    return model_poly


def classifySVMRbf(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    model_rbf = svm.SVC(C=10, kernel='rbf')
    model_rbf.fit(x, y)

    return model_rbf


def classifySVMSifmoid(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    model_sigmoid = svm.SVC(C=10, kernel='sigmoid')
    model_sigmoid.fit(x, y)

    return model_sigmoid


def regressionSVM(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    param = {'C': [x / 10 for x in range(1, 21, 2)],
             'kernel': ['poly', 'rbf', 'sigmoid']}
    gsearch = GridSearchCV(estimator=svm.SVR(C=10, kernel='poly')
                           , param_grid=param, cv=5, n_jobs=-1, scoring=regression.score())
    gsearch.fit(x.values, y)
    C = gsearch.best_params_['C']
    kernel = gsearch.best_params_['kernel']
    model = svm.SVR(C=C, kernel=kernel)
    model.fit(x, y)
    print(gsearch.best_params_)
    return model

def regressionSVMPoly(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    model_poly = svm.SVR(C=2, kernel='poly', degree=5)
    model_poly.fit(x, y)

    return model_poly


def regressionSVMRbf(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    model_rbf = svm.SVR(C=2, kernel='rbf')
    model_rbf.fit(x, y)

    return model_rbf


def regressionSVMSigmoid(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    model_sigmoid = svm.SVR(C=2, kernel='sigmoid')
    model_sigmoid.fit(x, y)
    return model_sigmoid


if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    classify_model = classifySVMPoly(df, feature_index)
    # regression_model = regressionSVM(df, feature_index)

    # df = pd.read_csv('../../error/source/test_norm.csv')
    # df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    # df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    # df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    # df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    # classifyTest(df, feature_index, classify_model)
