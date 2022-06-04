from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from error.data_process import processData
import numpy as np
from models import predict_model, mlp, svm, dt, rf

def mape_score(y, y_pre):
    return metrics.mean_absolute_percentage_error(y, y_pre)


def score():
    return metrics.make_scorer(mape_score, greater_is_better=False)

def evaluation(y, y_pre):
    """
    回归模型评测
    :param y: list 准确值 一维
    :param y_pre: list 预测值 一维
    :return: acc r2
    """
    y = np.array(y).reshape([-1, 1])
    y_pre = np.array(y_pre).reshape([-1, 1])
    r2 = metrics.r2_score(y, y_pre)

    acc = metrics.mean_absolute_percentage_error(y, y_pre)

    return acc * 100, r2 * 100



def sel_res():
    res_r = {r'$\chi^2$':['mue_ED', 'NMED', 'mue_ED0'],
             r'$\sigma^2$':['var_ED', 'var_RED', 'mue_RED'],
             r'$mr_c$':['WCRE','WCE','mue_ED0'],
             r'$mr_{Dd}$':['single-sided', 'ER', 'zero-error'],
             r'$mr_{Dq}$':['zero-error', 'single-sided', 'WCRE'],
             r'$l_{svm}$':['WCE', 'ER', 'mue_ARED'],
             r'$l_{dt}$':['mue_ED0', 'mue_ED', 'ER'],
             r'$l_{rf}$':['mue_ED0', 'var_ED0', 'mue_ARED'],
             r'$l_{mlp}$':['var_RED', 'mue_ED', 'mue_ED0'],
             'dfr': ['mue_ED', 'RMS_RED', 'mue_ED0']
             }
    return res_r

def regressionDraw(df, savename):
    """
    绘制domain以及spe回归模型指标图
    :param df:
    :param savename:
    :return:
    """
    plt.style.use(['science', 'ieee'])
    df = df.sort_values(by='MAPE', ascending=True)


    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    # plt.legend(label)
    plt.legend(loc='best')
    plt.savefig('../result/reg_evaluation_feature_sel/reg_' + savename + '_sel.pdf', bbox_inches='tight')
    df.to_csv('../result/reg_evaluation_feature_sel/reg_' + savename + '_sel.csv')
    plt.show()


def evaluationModelRegression(feature_index, model):
    """
    模型预测以及评价
    :param feature_index: 模型训练的特征
    :param model: 训练好的模型
    :return: 回归模型评测指标
    """
    df_test = pd.read_csv('../../error/source/test_norm.csv')
    df_test = processData(df_test.copy())
    y, y_pre = predict_model.predictRegression(model, feature_index, df_test)
    y = np.array(y)
    result = evaluation(y, y_pre)

    return result

if __name__ == '__main__':
    res_r = sel_res()
    df = pd.read_csv('../../error/source/train_norm.csv')
    dt_df = pd.DataFrame(index=res_r.keys(), columns=['MAPE', r'$R^2$'])
    svm_df = pd.DataFrame(index=res_r.keys(), columns=['MAPE', r'$R^2$'])
    rf_df = pd.DataFrame(index=res_r.keys(), columns=['MAPE', r'$R^2$'])
    mlp_df = pd.DataFrame(index=res_r.keys(), columns=['MAPE', r'$R^2$'])

    df_dict = {
                'dt': dt_df,
               'svm': svm_df,
               'mlp': mlp_df,
               'rf':rf_df
    }
    df = processData(df)

    # 添加模型
    func_dict = {}
    # func_dict['dt'] = dt.regressionDecisionTree
    # func_dict['rf'] = rf.regressionRF
    # func_dict['svm'] = svm.regressionSVM
    func_dict['mlp'] = mlp.regressionMLP

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

    # print(df_dict)

    for key in func_dict:
        regressionDraw(df_dict[key], key)
    # regressionDraw(dt_df, 'reg_dt' + '_sel_reg.pdf')