import pandas as pd
from sklearn import metrics
import sys
sys.path.append('..')
from error.data_process import processData
import numpy as np
from models import svm, dt, rf, predict_model, mlp


def evaluation(y, y_pre):
    top1 = metrics.accuracy_score(y, y_pre[:, 0])

    count = 0
    for i in range(len(y_pre)):
        if y[i] == y_pre[i][0] or y[i] == y_pre[i][1]:
            count += 1

    top2 = count / len(y_pre)

    y_recall_top1 = y.copy()
    y_pre_recall_top1 = y_pre[:, 0].copy()
    y_recall_top1[y_recall_top1 == 2] = 1
    y_pre_recall_top1[y_pre_recall_top1 == 2] = 1

    recall_1 = metrics.recall_score(y_recall_top1, y_pre_recall_top1, pos_label=0)

    weight_tpr = metrics.recall_score(y, y_pre[:, 0], average='weighted')
    macro_tpr = metrics.recall_score(y, y_pre[:, 0], average='macro')

    return top1, top2, recall_1, weight_tpr, macro_tpr



def sel_res():
    res_c = {'chi':[],
           'var':[],
           'mrmr_c':[],
           'mrmr_dd':[],
           'mrmr_dq':[],
           'lvm_svm':['var_ED', 'ER', 'WCE', 'WCRE', 'mue_RED', 'RMS_ED', 'RMS_RED', 'mue_ARED', 'single-sided', 'mue_ED', 'var_ARED', 'mue_ED0', 'var_RED'],
           'lvm_dt':['var_RED', 'single-sided', 'var_ARED', 'mue_ED0', 'ER', 'mue_ARED', 'mue_RED', 'var_ED', 'RMS_ED', 'NMED', 'WCE'],
           'lvm_rf':['var_ED0', 'WCE', 'mue_ED0', 'var_RED', 'RMS_RED'],
           'lvm_mlp':['NMED', 'RMS_RED', 'mue_ED0', 'WCE', 'mue_ARED', 'var_ED0', 'RMS_ED', 'zero-error', 'ER', 'mue_RED', 'mue_ED', 'single-sided'],
            'dfr':['mue_ED0', 'mue_ED', 'ER']
           }

    res_r = {'chi':['mue_ED', 'NMED', 'mue_ED0'],
           'var':['var_ED', 'var_RED', 'mue_RED'],
           'mrmr_c':['WCRE','WCE','mue_ED0'],
           'mrmr_dd':['single-sided', 'ER', 'zero-error'],
           'mrmr_dq':['zero-error', 'single-sided', 'WCRE'],
           'lvm_svm':['WCE', 'var_ED', 'NMED', 'var_ARED', 'mue_ARED', 'zero-error', 'mue_ED0', 'WCRE', 'mue_ED', 'RMS_ED', 'single-sided', 'var_RED', 'RMS_RED', 'mue_RED', 'ER', 'var_ED0'],
           'lvm_dt':['WCE', 'mue_ARED', 'mue_ED', 'NMED', 'ER', 'mue_ED0', 'mue_RED', 'RMS_ED', 'WCRE', 'var_RED', 'var_ARED', 'var_ED0', 'zero-error'],
           'lvm_rf':['var_ARED', 'RMS_ED', 'zero-error', 'mue_ED0', 'WCRE', 'var_ED0', 'RMS_RED', 'var_ED', 'var_RED', 'WCE', 'mue_ED', 'mue_ARED', 'single-sided'],
           'lvm_mlp':['WCE', 'var_ED', 'NMED', 'var_ARED', 'mue_ARED', 'zero-error', 'mue_ED0', 'WCRE', 'mue_ED', 'RMS_ED', 'single-sided', 'var_RED', 'RMS_RED', 'mue_RED', 'ER', 'var_ED0'],
            'dfr':['mue_ED0', 'mue_ED', 'ER']
           }
    return res_c, res_r

def evaluationModelClassify(feature_sel, model, model_name):

    y, y_pre = predict_model.predictClassify(model, feature_sel, model_name)
    y = np.array(y)
    result =evaluation(y, y_pre)

    return result

if __name__ == '__main__':
    res_c, res_r = sel_res()
    df = pd.read_csv('../../error/source/train_norm.csv')
    dt_df = pd.DataFrame(index=res_c.keys(), columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    df = processData(df)

    # 添加模型
    func_dict = {}
    func_dict['dt'] = dt.regressionDecisionTree
    # func_dict['svm'] = svm.regressionSVM
    # func_dict['rf'] = rf.regressionRF
    # func_dict['mlp'] = mlp.regressionMLP

    for key in res_r:
        feature_sel = res_r[key]
        df_sel = df.loc[:, feature_sel]

        for func in func_dict:
            acc = evaluationModelClassify(feature_sel, func_dict[func], func)
            print(acc)
