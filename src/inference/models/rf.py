from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import sys
sys.path.append('..')
sys.path.append('../../')
import matplotlib.pyplot as plt
import predict_model
from evaluate import classify, regression


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



def plotRF(df, savename):
    """
    绘制df数据折线图
    :param df: DataFrame 数据结构，
    :param savename:保存pdf的文件名
    """
    plt.style.use(['science', 'ieee'])
    df.to_csv('../result/csv/' + savename + '.csv')

    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    plt.legend(loc='best')
    plt.xticks(rotation=300)

    plt.savefig('../result/' + savename + '.pdf', bbox_inches='tight')
    plt.show()



# def getProb():
#     """
#     得到预测文件值
#     :return:
#     """
#     feature_sel = ['WCRE','WCE','mue_ED0']
#     fixed_feature = ['net', 'dataset', 'concat']
#     feature_sel += fixed_feature
#     df1 = pd.read_csv('../../error/source/train_norm.csv', index_col='mul_name')
#     df1 = processData(df1)
#     df2 = pd.read_csv('../../error/source/test_norm.csv', index_col='mul_name')
#
#     model = classifyRF(df1, feature_sel)
#     y, y_pre = predict_model.predictClassify(model, feature_sel, 'rf', True)
#     df2.insert(5, column='y_pre', value=y_pre[:, 0])
#     # df2.loc[:, 'y_pre'] = y_pre[:, 0]
#     df2.sort_values(by=['y_pre', 'untrained_acc'], inplace=True, ascending=[True, False])
#     df2.to_csv('../result/csv/cla_pre.csv')
#



def buildErrorModel():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')

    # 构建分类误差模型
    feature_index = ['WCRE', 'WCE', 'mue_ED0']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist',  'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    predict_model.claErrorModel(df_train, df_test, feature_index, indexes, classifyRF, 'rf', dt_df, 'cla_rf_model')

    # 构建回归误差模型
    feature_index = ['var_ED', 'var_RED', 'mue_RED']
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