from sklearn import tree
import pandas as pd


def classifyDecisionTree(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]

    tree_model = tree.DecisionTreeClassifier(criterion='gini')
    tree_model.fit(x, y)

    return tree_model


def regressionDecisionTree(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    tree_model = tree.DecisionTreeRegressor(criterion='gini')
    tree_model.fit(x, y)

    return tree_model


if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    classifyDecisionTree(df, feature_index)
