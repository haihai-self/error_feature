from sklearn import svm
import pandas as pd
from sklearn.model_selection import GridSearchCV
import sys

sys.path.append('..')
from evaluate import regression, classify
import predict_model

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


def buildErrorModel():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')

    # 构建分类误差模型
    feature_index = ['mue_ED0', 'mue_ED', 'ER']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist',  'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    predict_model.claErrorModel(df_train, df_test, feature_index, indexes, classifySVM, 'svm', dt_df, 'cla_svm_model')

    # # 构建回归误差模型
    # feature_index =['mue_ED0', 'mue_ED', 'ER']
    # dt_df = pd.DataFrame(index=indexes, columns=['MAPE', r'$\chi^2$'])
    # predict_model.regErrorModel(df_train, df_test, feature_index, indexes, regressionRF, 'rf', dt_df, 'reg_rf_model')

    # 构建retrain分类误差模型
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')


    feature_index = ['WCRE', 'WCE', 'mue_ED0']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist',  'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])


if __name__ == '__main__':
    buildErrorModel()
