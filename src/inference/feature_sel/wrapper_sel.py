import pandas as pd
import random
import sys
sys.path.append('..')
from models import svm, dt, rf, predict_model
from evaluate import classify
import numpy as np

def evaluationModel(df_train, feature_index, model, model_name):
    df_test = pd.read_csv('../../error/source/test_norm.csv')
    df_test.loc[df_test.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df_test.loc[df_test.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df_test.loc[df_test.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df_test.loc[df_test.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1

    y, y_pre = predict_model.predictClassify(model, feature_index, model_name)
    y = np.array(y)
    result = classify.evaluation(y, y_pre)

    return result

def lvm(df, func, model_name):
    e = [0]
    feature_index = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                     'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    d = len(feature_index)
    a = feature_index
    t = 0
    T = 30
    while t < T:
        d_cur = random.randint(1, len(feature_index))
        a_cur = random.sample(feature_index, d_cur)
        model = func(df, a_cur)
        e_cur = evaluationModel(df, a_cur, model, model_name)
        if e_cur[-1] > e[-1] or (e_cur[-1] == e[-1] and d_cur < d):
            t = 0
            e = e_cur
            d = d_cur
            a = a_cur
        else:
            t+=1
    return a, e

if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1
    # func_dict = {'dt': dt.classifyDecisionTree, 'svm':svm.classifySVM, 'rf': rf.classifyRF}
    func_dict = {'dt': dt.classifyDecisionTree, 'svm':svm.classifySVM}

    for key in func_dict:
        feature = lvm(df, func_dict[key], key)
        print(feature)
