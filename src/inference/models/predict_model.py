from sklearn import metrics
import numpy as np
import pandas as pd
from error.data_process import processData, processDataSpec

def predictClassify(model, feature_index, model_name):
    # 读取测试文件
    df = pd.read_csv('../../error/source/test_norm.csv')
    df = processData(df)
    x = df.loc[:, feature_index]
    y = df.loc[:, 'classify']

    # 计算前两类预测值
    # x = model._validate_X_predict(x, True)
    if model_name == 'svm':
        proba = model.decision_function(x)
    elif model_name == 'mlp':
        proba = model.predict(x)
    else:
        proba = model.predict_proba(x)
    top1 = np.argmax(proba, axis=1)
    for i in range(len(proba)):
        proba[i][top1[i]] = -1
    top2 = np.argmax(proba, axis=1)
    y_pre = np.stack([top1, top2], axis=1)

    return y, y_pre

def predictSpecClassify(model, feature_index, model_name, spec):
    # 读取测试文件
    df = pd.read_csv('../../error/source/test_norm.csv')
    df = processDataSpec(df)

    x = df.loc[df['concat'] == spec, feature_index]
    y = df.loc[df['concat'] == spec, 'classify']
    y = np.array(y)

    # 计算前两类预测值
    # x = model._validate_X_predict(x, True)
    if model_name == 'svm':
        proba = model.decision_function(x)
    elif model_name == 'mlp':
        proba = model.predict(x)
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
    df = processData(df)
    x = df.loc[:, feature_index]
    y = df.loc[:, 'untrained_acc']

    y_pre = model.predict(x)

    return y, y_pre

def predictSpectRegression(model, feature_index, model_name, spec):
    # 读取测试文件
    df = pd.read_csv('../../error/source/test_norm.csv')
    df = processDataSpec(df)

    x = df.loc[df['concat'] == spec, feature_index]
    y = df.loc[df['concat'] == spec, 'untrained_acc']

    y_pre = model.predict(x)
    y = np.array(y)

    return y, y_pre