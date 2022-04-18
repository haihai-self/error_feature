from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from evaluate import classify, regression
from sklearn import metrics
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from models.predict_model import predictClassify, predictSpecClassify, predictRegression, predictSpectRegression
from error.data_process import processDataSpec, processData
from evaluate import classify, regression


def classifySVM(df, feature_index):
    """
    svm 分类模型，需要搜索
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    # svm分类模型， 搜素最优参数
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    param = {'C': [x for x in range(400, 1000, 8)],
             'kernel': ['rbf']}
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
    """
    # svm poly分类模型
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    model_poly = svm.SVC(C=10, kernel='poly', degree=5)
    model_poly.fit(x, y)

    return model_poly


def classifySVMRbf(df, feature_index):
    """
    # svm rbf 分类模型
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    model_rbf = svm.SVC(C=10, kernel='rbf')
    model_rbf.fit(x, y)

    return model_rbf


def classifySVMSifmoid(df, feature_index):
    """
    # svm sigmoid 分类模型
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    model_sigmoid = svm.SVC(C=10, kernel='sigmoid')
    model_sigmoid.fit(x, y)

    return model_sigmoid


def regressionSVM(df, feature_index):
    """
    # svm 回归最优模型， 搜索最优参数
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    param = {'C': [x for x in range(20, 31, 2)],
             'kernel': ['rbf']}
    gsearch = GridSearchCV(estimator=svm.SVR(C=5, kernel='poly')
                           , param_grid=param, cv=5, n_jobs=-1, scoring=regression.score())
    gsearch.fit(x.values, y)
    kernel = gsearch.best_params_['kernel']
    model = svm.SVR(C=5, kernel=kernel)
    model.fit(x, y)
    print(gsearch.best_params_)
    return model


def regressionSVMPoly(df, feature_index):
    """
    # svm poly 回归最优模型
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    model_poly = svm.SVR(C=2, kernel='poly', degree=5)
    model_poly.fit(x, y)

    return model_poly


def regressionSVMRbf(df, feature_index):
    """
    # svm rbf 回归最优模型
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    model_rbf = svm.SVR(C=2, kernel='rbf')
    model_rbf.fit(x, y)

    return model_rbf


def regressionSVMSigmoid(df, feature_index):
    """
    # svm sigmoid 回归最优模型
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: svm model
    """
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    model_sigmoid = svm.SVR(C=2, kernel='sigmoid')
    model_sigmoid.fit(x, y)
    return model_sigmoid


def plotSVM(df, savename):
    """
    绘制df数据折线图
    :param df: DataFrame 数据结构，
    :param savename:保存pdf的文件名
    """
    plt.style.use(['science', 'ieee'])
    # df = df.sort_values(by='mape', ascending=True)
    df.to_csv('../result/csv/' + savename + '.csv')
    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    # plt.legend(label)
    plt.legend(loc='best')
    plt.xticks(rotation=300)

    plt.savefig('../result/' + savename + '.pdf', bbox_inches='tight')
    # plt.show()


def svmClaErrorModel():
    # 训练svm分类误差模型，并且绘制对应指标图
    df = pd.read_csv('../../error/source/train_norm.csv')
    df = processDataSpec(df)
    # feature_index = ['WCRE', 'WCE', 'mue_ED0']
    feature_index = ['mue_ED0', 'mue_ED', 'ER']
    # feature_index = ['var_ED', 'var_RED', 'mue_RED']

    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])

    for index in indexes:
        if index == 'domain':
            fixed_feature = ['net', 'dataset', 'concat']
            df_train = processData(df)
            model = classifySVM(df_train, feature_index + fixed_feature)
            y, y_pre = predictClassify(model, feature_index + fixed_feature, 'svm')

        else:

            df_train = df[df['concat'] == index]

            model = classifySVM(df_train, feature_index)
            y, y_pre = predictSpecClassify(model, feature_index, 'svm', index)
        res = classify.evaluation(y, y_pre)
        dt_df.loc[index, :] = res
    plotSVM(dt_df, 'cla_svm_model')


if __name__ == '__main__':
    svmClaErrorModel()
