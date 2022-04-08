from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd


def var_sel(df, ):
    des = df.describe().T.rename(columns={'std': 'std_var'})
    var = des.loc[:, 'std_var'].sort_values(ascending=False)
    return var


def chi2_sel(df):
    model = SelectKBest(chi2, k=2)
    data = df.loc[:,
           ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
            'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE']].astype(float)
    target = df.loc[:, 'untrained']
    target[target > 0.7] = 1
    target[target < 0.7] = 3
    target.astype(int)

    model.fit(data,target)
    score = -np.log(model.pvalues_)

    se = pd.Series(index=['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
            'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE'],
                   data = score)
    return se


def mRMR_sel():


if __name__ == '__main__':
    df = pd.read_csv('../error/source/train_norm.csv', index_col='mul_name')
    var_sel(df)
    chi2_sel(df)

