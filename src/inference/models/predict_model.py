from sklearn import metrics
import numpy as np
import pandas as pd

def predictClassify(model, feature_index, model_name):
    # 读取测试文件
    df = pd.read_csv('../../error/source/test_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    x = df.loc[:, feature_index]
    y = df.loc[:, 'classify']

    # 计算前两类预测值
    # x = model._validate_X_predict(x, True)
    if model_name == 'svm':
        proba = model.decision_function(x)
    else:
        proba = model.predict_proba(x)
    top1 = np.argmax(proba, axis=1)
    for i in range(len(proba)):
        proba[i][top1[i]] = -1
    top2 = np.argmax(proba, axis=1)
    y_pre = np.stack([top1, top2], axis=1)

    return y, y_pre


def predictRegression(model, feature_index):
    # 读取测试文件
    df = pd.read_csv('../../error/source/test_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    x = df.loc[:, feature_index]
    y = df.loc[:, 'untrained_acc']

    y_pre = model.predict(x)

    return y, y_pre

