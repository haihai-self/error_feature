from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, mutual_info_regression
from sklearn import metrics
import numpy as np
import pandas as pd
import pymrmr


def varSel(df, ):
    des = df.describe().T.rename(columns={'std': 'std_var'})
    var = des.loc[:, 'std_var'].sort_values(ascending=False)
    return var


def chi2Sel(df):
    model = SelectKBest(chi2, k=14)
    data = df.loc[:,
           ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
            'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE']].astype(float)
    target = df.loc[:, 'classify']

    target.astype(int)

    model.fit(data, target)
    score = -np.log(model.pvalues_)
    print(score)
    se = pd.Series(index=['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
                          'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE'],
                   data=score)
    return se


def regressionMrmrSel(df):
    d = df.loc[:,
        ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
         'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']].astype(float)
    f = df.loc[:, 'untrained_acc']

    # d = np.array(d)
    # f = np.array(f)
    def getmultimi(da, dt):
        c = []
        for i in range(len(da[1])):
            c.append(metrics.normalized_mutual_info_score(da[:, i], dt))
        return c

    # 调库计算
    # df = pd.merge(f,d, left_index=True, right_index=True)
    # df = df.sample(frac=0.2)
    # a = pymrmr.mRMR(df, 'MID',14)
    # print(a)

    # 计算与label的互信息
    mi_y = mutual_info_regression(d, f)

    # 计算特征之间的互信息平均值
    mi_x = []
    for index, data in d.iteritems():
        target = d.loc[:, index]
        input = d.drop(columns=index)
        temp = mutual_info_regression(input, target)
        mi_x.append(temp.sum() / input.shape[1])
    # 计算phi得分
    score = mi_y - mi_x
    se = pd.Series(index=['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
         'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error'],
                   data=score)
    # idxs = sorted(range(len(score)), key=lambda k: score[k], reverse=True)  # 得到与之对应的idx
    # score = sorted(score, reverse=True)  # 互信息排序

    return se


def classifyMrmrSel(df):
    d = df.loc[:,
        ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
         'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']].astype(float)
    f = df.loc[:, 'untrained_acc']

if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1

    var = varSel(df)
    chi2 = chi2Sel(df)
    re_mrmr = regressionMrmrSel(df)
