from sklearn import svm
import pandas as pd
import numpy as np
from multiprocessing import Pool


def classifySVM(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    model_poly = svm.SVC(C=10, kernel='poly', degree=5)
    model_poly.fit(x, y)

    model_rbf = svm.SVC(C=10, kernel='rbf')
    model_rbf.fit(x, y)

    model_sigmoid = svm.SVC(C=10, kernel='sigmoid')
    model_sigmoid.fit(x, y)

    return model_poly, model_rbf, model_sigmoid


def regressionSVM(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    model_poly = svm.SVR(C=10, kernel='poly', degree=5)
    model_poly.fit(x, y)

    model_rbf = svm.SVR(C=10, kernel='rbf')
    model_rbf.fit(x, y)

    model_sigmoid = svm.SVR(C=10, kernel='sigmoid')
    model_sigmoid.fit(x, y)
    return model_poly, model_rbf, model_sigmoid


def classifyTest(df, feature_index, models):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]
    y_pre = []
    for model in models:
        t = model.predict(x)
        print(t)
        y_pre.append(t)
    count = 0


if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    classify_model = classifySVM(df, feature_index)
    regression_model = regressionSVM(df, feature_index)

    df = pd.read_csv('../../error/source/test_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    classifyTest(df, feature_index, classify_model)
