import pandas as pd
import random
import sys

sys.path.append('..')
sys.path.append('../../')
sys.path.append('.')
from models import svm, dt, rf, predict_model, mlp
from evaluate import classify, regression
import numpy as np
from error.data_process import processData


def evaluationModelClassify(feature_index, model, model_name):
    """
    分类模型评价
    :param feature_index: 模型训练使用的特征
    :param model: 需要测试的模型
    :param model_name: 模型对应的名称 SVM DT MLP RF等
    :return: 分类模型测试结果 top1 top2 recall-1 weight-tpr macro-tpr
    """
    df_test = pd.read_csv('../../error/source/test_norm.csv')
    df_test = processData(df_test)
    y, y_pre = predict_model.predictClassify(model, feature_index, model_name, df_test)
    y = np.array(y)
    result = classify.evaluation(y, y_pre)

    return result


def lvmClassify(df, func, model_name):
    """
    LVM框架对模型进行特征筛选
    :param df: 需要训练你的数据
    :param func: 对应的模型
    :param model_name: 模型名称SVM DT MLP RF
    :return: 两个list list1 筛选出的特征， list2 对应模型的评价结果
    """
    e = [0]
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    fixed_feature = ['net', 'dataset', 'concat']
    d = len(feature_index)
    a = feature_index
    t = 0
    T = 30
    while t < T:
        # d_cur = random.randint(1, len(feature_index))
        d_cur = 3
        a_cur = random.sample(feature_index, d_cur)
        model = func(df, a_cur + fixed_feature)
        e_cur = evaluationModelClassify(a_cur + fixed_feature, model, model_name)
        if e_cur[-1] > e[-1] or (e_cur[-1] == e[-1] and d_cur < d):
            t = 0
            e = e_cur
            d = d_cur
            a = a_cur
        else:
            t += 1
        print(t, e_cur)
    return a, e


def evaluationModelRegression(feature_index, model):
    """
    回归模型评价
    :param feature_index:训练使用的特征
    :param model:已经训练好的模型
    :return: list acc r2
    """
    y, y_pre = predict_model.predictRegression(model, feature_index)
    y = np.array(y)
    result = regression.evaluation(y, y_pre)

    return result

def lvmRegression(df, func):
    """
    LVM回归模型特征筛选
    :param df: 需要训练你的数据 DataFrame
    :param func: 模型名称 SVM MLP DT RF
    :return: 2个list list1 筛选出的特征 list2 回归模型评价指标
    """
    e = [1000]
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    fixed_feature = ['net', 'dataset', 'concat']
    d = len(feature_index)
    a = feature_index
    t = 0
    T = 30
    while t < T:
        # d_cur = random.randint(1, len(feature_index))
        d_cur = 3

        a_cur = random.sample(feature_index, d_cur)
        model = func(df, a_cur + fixed_feature)
        e_cur = evaluationModelRegression(a_cur + fixed_feature, model)
        if e_cur[0] < e[0] or (e_cur[0] == e[0] and d_cur < d):
            t = 0
            e = e_cur
            d = d_cur
            a = a_cur
        else:
            t += 1
        print(t, e_cur)

    return a, e
if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df = processData(df)

    # 分类问题模型
    func_dict = {}
    # func_dict['dt'] = dt.classifyDecisionTree
    # func_dict['rf'] = rf.classifyRF
    # func_dict['svm'] = svm.classifySVM
    func_dict['mlp'] = mlp.classifyMLP
    feature = {}

    for key in func_dict:
        temp = lvmClassify(df, func_dict[key], key)
        feature[key] = temp
    print("\n\n the result of lvm: \n")
    for key in feature:
        print(key, ':', feature[key])

    # 回顾问题模型
    # func_dict = {}
    # func_dict['dt'] = dt.regressionDecisionTree
    # func_dict['svm'] = svm.regressionSVM
    # func_dict['rf'] = rf.regressionRF
    # func_dict['mlp'] = mlp.regressionMLP

    # for key in func_dict:
    #     feature = lvmRegression(df, func_dict[key])
    #     print(feature)