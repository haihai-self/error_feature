from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics
import pandas as pd


def classifyRF(df, feature_index):
    y = df.loc[:, 'classify']
    x = df.loc[:, feature_index]
    param = {'n_estimators': range(10, 201, 10)}
    gsearch = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60, oob_score=True, random_state=10)
                           , param_grid=param, cv=5, n_jobs=-1)
    gsearch.fit(x.values, y)
    # print( gsearch.best_params_, gsearch.best_score_)
    model = gsearch.best_estimator_
    # model = RandomForestClassifier(n_estimators=, oob_score=True, random_state=10)
    model.fit(x, y)
    return model


def regressionRF(df, feature_index):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_index]

    param = {'n_estimators': range(10, 201, 10)}
    gsearch = GridSearchCV(estimator=RandomForestRegressor(n_estimators=60, oob_score=True, random_state=10)
                           , param_grid=param, cv=5, n_jobs=-1)
    gsearch.fit(x.values, y)
    # print( gsearch.best_params_, gsearch.best_score_)
    model = gsearch.best_estimator_
    # model = RandomForestClassifier(n_estimators=, oob_score=True, random_state=10)
    model.fit(x, y)
    return model


if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']

    classifyRF(df, feature_index)
    regressionRF(df, feature_index)
