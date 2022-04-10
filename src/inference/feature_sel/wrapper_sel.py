import pandas as pd
import random
import sys
sys.path.append('..')
from ..models import svm, dt, rf

def evaluationModel(df_train, feature_index, model):
    df_test = pd.read_csv('../../error/source/test_norm.csv')
    df_test.loc[df_test.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df_test.loc[df_test.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df_test.loc[df_test.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df_test.loc[df_test.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1


    return

def lvm(df, ):
    e = 10000
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    d = len(feature_index)
    a = feature_index
    t = 0
    T = 15
    while t < T:
        d_cur = random.randint(1, len(feature_index))
        a_cur = random.sample(feature_index, d_cur)
        e_cur = 0
        if e_cur < e or (e_cur == e and d_cur < d):
            t = 0
            e = e_cur
            d = d_cur
            a = a_cur
        else:
            t+=1
    return a

if __name__ == '__main__':
    lvm()
