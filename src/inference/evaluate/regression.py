from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from error.data_process import processData
import numpy as np
from models import svm, dt, rf, predict_model, mlp

def mape_score(y, y_pre):
    return metrics.mean_absolute_percentage_error(y, y_pre)


def score():
    return metrics.make_scorer(mape_score, greater_is_better=False)

def evaluation(y, y_pre):
    y = np.array(y).reshape([-1, 1])
    y_pre = np.array(y_pre).reshape([-1, 1])
    r2 = metrics.r2_score(y, y_pre)

    acc = metrics.mean_absolute_percentage_error(y, y_pre)

    return acc * 100, r2 * 100



def sel_res():

    res_c = {r'$\chi^2$':['mue_ED', 'NMED', 'mue_ED0'],
             'var':['var_ED', 'var_RED', 'mue_RED'],
             r'$mr_c$':['WCRE','WCE','mue_ED0'],
             r'$mr_{Dd}$':['single-sided', 'ER', 'zero-error'],
             r'$mr_{Dq}$':['zero-error', 'single-sided', 'WCRE'],
             r'$l_{svm}$':['var_ED', 'ER', 'WCE', 'WCRE', 'mue_RED', 'RMS_ED', 'RMS_RED', 'mue_ARED', 'single-sided', 'mue_ED', 'var_ARED', 'mue_ED0', 'var_RED'],
             r'$l_{dt}$':['var_RED', 'single-sided', 'var_ARED', 'mue_ED0', 'ER', 'mue_ARED', 'mue_RED', 'var_ED', 'RMS_ED', 'NMED', 'WCE'],
             r'$l_{rf}$':['var_ED0', 'WCE', 'mue_ED0', 'var_RED', 'RMS_RED'],
             r'$l_{mlp}$':['NMED', 'RMS_RED', 'mue_ED0', 'WCE', 'mue_ARED', 'var_ED0', 'RMS_ED', 'zero-error', 'ER', 'mue_RED', 'mue_ED', 'single-sided'],
             'dfr':['mue_ED0', 'mue_ED', 'ER']
             }
    res_r = {r'$\chi^2$':['mue_ED', 'NMED', 'mue_ED0'],
             'var':['var_ED', 'var_RED', 'mue_RED'],
             r'$mr_c$':['WCRE','WCE','mue_ED0'],
             r'$mr_{Dd}$':['single-sided', 'ER', 'zero-error'],
             r'$mr_{Dq}$':['zero-error', 'single-sided', 'WCRE'],
             r'$l_{svm}$':['WCE', 'var_ED', 'NMED', 'var_ARED', 'mue_ARED', 'zero-error', 'mue_ED0', 'WCRE', 'mue_ED', 'RMS_ED', 'single-sided', 'var_RED', 'RMS_RED', 'mue_RED', 'ER', 'var_ED0'],
             r'$l_{dt}$':['WCE', 'mue_ARED', 'mue_ED', 'NMED', 'ER', 'mue_ED0', 'mue_RED', 'RMS_ED', 'WCRE', 'var_RED', 'var_ARED', 'var_ED0', 'zero-error'],
             r'$l_{rf}$':['var_ARED', 'RMS_ED', 'zero-error', 'mue_ED0', 'WCRE', 'var_ED0', 'RMS_RED', 'var_ED', 'var_RED', 'WCE', 'mue_ED', 'mue_ARED', 'single-sided'],
             r'$l_{mlp}$':['WCE', 'var_ED', 'NMED', 'var_ARED', 'mue_ARED', 'zero-error', 'mue_ED0', 'WCRE', 'mue_ED', 'RMS_ED', 'single-sided', 'var_RED', 'RMS_RED', 'mue_RED', 'ER', 'var_ED0'],
             'dfr':['mue_ED0', 'mue_ED', 'ER']
             }
    return res_c, res_r

def regressionDraw(df, savename):
    plt.style.use(['science', 'ieee'])
    df = df.sort_values(by='mape', ascending=True)


    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    # plt.legend(label)
    plt.legend(loc='best')
    plt.savefig('../result/' + savename, bbox_inches='tight')
    plt.show()


def evaluationModelRegression(feature_index, model):

    y, y_pre = predict_model.predictRegression(model, feature_index)
    y = np.array(y)
    result = evaluation(y, y_pre)

    return result

if __name__ == '__main__':
    res_c, res_r = sel_res()
    df = pd.read_csv('../../error/source/train_norm.csv')
    dt_df = pd.DataFrame(index=res_r.keys(), columns=['mape', r'$R^2$'])
    svm_df = pd.DataFrame(index=res_r.keys(), columns=['mape', r'$R^2$'])
    rf_df = pd.DataFrame(index=res_r.keys(), columns=['mape', r'$R^2$'])
    mlp_df = pd.DataFrame(index=res_r.keys(), columns=['mape', r'$R^2$'])

    df_dict = {'dt': dt_df,
               'svm': svm_df,
               'mlp': mlp_df,
               'rf':rf_df}
    df = processData(df)

    # 添加模型
    func_dict = {}
    # func_dict['dt'] = dt.regressionDecisionTree
    func_dict['rf'] = rf.regressionRF
    # func_dict['svm'] = svm.regressionSVM
    # func_dict['mlp'] = mlp.regressionMLP

    fixed_feature = ['net', 'dataset', 'concat']
    count = 0
    for key in res_r:
        print(count)
        count += 1
        feature_sel = res_r[key]

        for func in func_dict:
            model = func_dict[func](df, feature_sel + fixed_feature)
            acc = evaluationModelRegression(feature_sel + fixed_feature, model)
            df_dict[func].loc[key, :] = acc

    for key in func_dict:
        regressionDraw(df_dict[key], 'reg_' + key+'_sel_reg.pdf')
    # regressionDraw(dt_df, 'reg_dt' + '_sel_reg.pdf')