from sklearn import tree
import pandas as pd
from models.predict_model import predictClassify, predictSpecClassify, predictRegression, predictSpectRegression
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
from error.data_process import processDataSpec, processData
from evaluate import classify, regression


def classifyDecisionTree(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    tree_model = tree.DecisionTreeClassifier(criterion='gini')
    tree_model.fit(x, y)

    return tree_model


def regressionDecisionTree(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    tree_model = tree.DecisionTreeRegressor()
    tree_model.fit(x, y)

    return tree_model

def plotDT(df, savename):
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

def dfClaErrorModel():
    df = pd.read_csv('../../error/source/train_norm.csv')
    df = processDataSpec(df)
    feature_index = ['WCRE', 'WCE', 'mue_ED0']
    # feature_index = ['mue_ED0', 'mue_ED', 'ER']
    # feature_index = ['var_ED', 'var_RED', 'mue_RED']

    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])

    for index in indexes:
        if index == 'domain':
            fixed_feature = ['net', 'dataset', 'concat']
            df_train = processData(df)
            model = classifyDecisionTree(df_train, feature_index + fixed_feature)
            y, y_pre = predictClassify(model, feature_index + fixed_feature, index)

        else:

            df_train = df[df['concat'] == index]

            model = classifyDecisionTree(df_train, feature_index)
            y, y_pre = predictSpecClassify(model, feature_index, 'dt', index)
        res = classify.evaluation(y, y_pre)
        dt_df.loc[index, :] = res
    plotDT(dt_df, 'cla_dt_model')

def dfRegErrorModel():
    df = pd.read_csv('../../error/source/train_norm.csv')
    df = processDataSpec(df)
    feature_index = ['var_ED', 'var_RED', 'mue_RED']
    # feature_index = ['mue_ED0', 'mue_ED', 'ER']

    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['MAPE', r'$\chi^2$'])

    for index in indexes:
        if index == 'domain':
            fixed_feature = ['net', 'dataset', 'concat']
            df_train = processData(df)
            model = regressionDecisionTree(df_train, feature_index + fixed_feature)
            y, y_pre = predictRegression(model, feature_index + fixed_feature)

        else:

            df_train = df[df['concat'] == index]

            model = regressionDecisionTree(df_train, feature_index)
            y, y_pre = predictSpectRegression(model, feature_index, 'dt', index)
        res = regression.evaluation(y, y_pre)
        dt_df.loc[index, :] = res
    plotDT(dt_df, 'reg_dt_model')

if __name__ == '__main__':
    dfRegErrorModel()
    dfClaErrorModel()
