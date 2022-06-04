from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import sys
sys.path.append('..')
sys.path.append('../../')
import matplotlib.pyplot as plt
from evaluate import classify, regression
from models import predict_model

def classifyRF(df, feature_index):
    """
    RF分类模型，搜索最优C
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: RF model
    """

    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]
    param = {'n_estimators': range(60, 400, 20)}
    gsearch = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60, oob_score=True, random_state=10)
                           , param_grid=param, cv=5, n_jobs=-1, scoring=classify.score())
    gsearch.fit(x.values, y)
    print(gsearch.best_params_)
    n_estimators = gsearch.best_params_['n_estimators']
    model = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=10)
    # model = RandomForestClassifier(n_estimators=, oob_score=True, random_state=10)
    model.fit(x, y)
    # print(model.get_params())
    # model.max_depth
    return model


def regressionRF(df, feature_index):
    """
    RF回归模型，搜索最优C
    :param df: train data DataFrame 数据结构
    :param feature_index: list 需要用到的feature
    :return: RF model
    """
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    param = {'n_estimators': range(10, 201, 10)}
    gsearch = GridSearchCV(estimator=RandomForestRegressor(n_estimators=60, random_state=10)
                           , param_grid=param, cv=5, n_jobs=-1, scoring=regression.score())
    gsearch.fit(x.values, y)
    print(gsearch.best_params_)
    model = gsearch.best_estimator_
    # model = RandomForestClassifier(n_estimators=, oob_score=True, random_state=10)
    model.fit(x, y)
    return model



def buildErrorModel():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')

    # 构建分类误差模型
    feature_index = ['var_ED0', 'var_RED', 'mue_ED0']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    predict_model.claErrorModel(df_train, df_test, feature_index, indexes, classifyRF, 'rf', dt_df, 'cla_rf_model')

    # 构建回归误差模型
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ARED']
    dt_df = pd.DataFrame(index=indexes, columns=['MAPE', r'$\chi^2$'])
    predict_model.regErrorModel(df_train, df_test, feature_index, indexes, regressionRF, 'rf', dt_df, 'reg_rf_model')

    # 构建retrain分类误差模型
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')


    feature_index = ['WCRE', 'WCE', 'mue_ED0']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist',  'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])

if __name__ == '__main__':
    buildErrorModel()
    # rfRegErrorModel()
    # getProb()
    # classifyRetrainRF()