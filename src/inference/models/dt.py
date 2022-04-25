from sklearn import tree
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
import predict_model

def classifyDecisionTree(df, feature_index):
    """
    使用制定的特征训练分类DT
    :param df: train data DataFrame数据类型
    :param feature_index: 选择训练的特征 list
    :return: DT分类模型
    """
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    tree_model = tree.DecisionTreeClassifier(criterion='gini')
    tree_model.fit(x, y)
    print(tree_model.tree_.node_count)

    return tree_model


def regressionDecisionTree(df, feature_index):
    """
    使用制定的特征训练回归DT
    :param df: train data DataFrame数据类型
    :param feature_index: 选择训练的特征 list
    :return: DT分类模型
    """
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    tree_model = tree.DecisionTreeRegressor()
    tree_model.fit(x, y)
    print(tree_model.tree_.node_count)

    return tree_model

def plotDT(df, savename):
    """
    绘制模型对应的指标图
    :param df: plot data DataFrame数据结构
    :param savename: 保存的名字, str
    :return:
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
    plt.show()


def buildErrorModel():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')

    # 构建分类误差模型
    feature_index = ['WCRE', 'WCE', 'mue_ED0']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist',  'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    predict_model.claErrorModel(df_train, df_test, feature_index, indexes, classifyDecisionTree, 'dt', dt_df, 'cla_dt_model')

    # 构建回归误差模型
    feature_index = ['var_ED', 'var_RED', 'mue_RED']
    dt_df = pd.DataFrame(index=indexes, columns=['MAPE', r'$\chi^2$'])
    predict_model.regErrorModel(df_train, df_test, feature_index, indexes, regressionDecisionTree, 'dt', dt_df, 'reg_dt_model')

    # 构建retrain分类误差模型
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')

if __name__ == '__main__':
    buildErrorModel()
