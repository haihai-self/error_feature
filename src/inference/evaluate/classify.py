import pandas as pd
from sklearn import metrics
import sys
sys.path.append('..')
from error.data_process import processData
import numpy as np
from models import svm, dt, rf, mlp
from models.predict_model import predictClassify
import matplotlib.pyplot as plt


def macro_tpr(y, y_pre):
    return metrics.recall_score(y, y_pre, average='macro')


def score():
    return metrics.make_scorer(macro_tpr, greater_is_better=True)

def evaluation(y, y_pre):
    """
    求出分类预测得到的对应的指标
    :param y: 真实标签，一维
    :param y_pre: 预测标签，二维，分别为top1预测概率以及top2预测概率
    :return: list top1 top2 recall-1 weight-tpr macro-tpr
    """

    # 计算top1
    top1 = metrics.accuracy_score(y, y_pre[:, 0])

    # 计算top2
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

    return top1 * 100, top2 * 100, recall_1 * 100, weight_tpr * 100, macro_tpr * 100



def sel_res():
    """
    特征排序的结果
    :return: 两个dict res_c 分类模型特征排序结果, res_r 回归模型排序结果
    """
    res_c = {
            r'$\chi^2$':['mue_ED', 'NMED', 'mue_ED0'],
             r'$\sigma^2$':['var_ED', 'var_RED', 'mue_RED'],
             r'$mr_c$':['WCRE','WCE','mue_ED0'],
             r'$mr_{Dd}$':['single-sided', 'ER', 'zero-error'],
             r'$mr_{dq}$':['zero-error', 'single-sided', 'WCRE'],
             r'$l_{svm}$':['var_ED', 'ER', 'WCE', 'WCRE', 'mue_RED', 'RMS_ED', 'RMS_RED', 'mue_ARED', 'single-sided', 'mue_ED', 'var_ARED', 'mue_ED0', 'var_RED'],
             r'$l_{dt}$':['var_RED', 'single-sided', 'var_ARED', 'mue_ED0', 'ER', 'mue_ARED', 'mue_RED', 'var_ED', 'RMS_ED', 'NMED', 'WCE'],
             r'$l_{rf}$':['var_ED0', 'WCE', 'mue_ED0', 'var_RED', 'RMS_RED'],
             r'$l_{mlp}$':['NMED', 'RMS_RED', 'mue_ED0', 'WCE', 'mue_ARED', 'var_ED0', 'RMS_ED', 'zero-error', 'ER', 'mue_RED', 'mue_ED', 'single-sided'],
             'dfr':['mue_ED0', 'mue_ED', 'ER']
             }
    res_r = {'chi':['mue_ED', 'NMED', 'mue_ED0'],
             'var':['var_ED', 'var_RED', 'mue_RED'],
             r'$mr_c$':['WCRE','WCE','mue_ED0'],
             r'$mr_{Dd}$':['single-sided', 'ER', 'zero-error'],
             r'$mr_{dq}$':['zero-error', 'single-sided', 'WCRE'],
             r'$l_{svm}$':['WCE', 'var_ED', 'NMED', 'var_ARED', 'mue_ARED', 'zero-error', 'mue_ED0', 'WCRE', 'mue_ED', 'RMS_ED', 'single-sided', 'var_RED', 'RMS_RED', 'mue_RED', 'ER', 'var_ED0'],
             r'$l_{dt}$':['WCE', 'mue_ARED', 'mue_ED', 'NMED', 'ER', 'mue_ED0', 'mue_RED', 'RMS_ED', 'WCRE', 'var_RED', 'var_ARED', 'var_ED0', 'zero-error'],
             r'$l_{rf}$':['var_ARED', 'RMS_ED', 'zero-error', 'mue_ED0', 'WCRE', 'var_ED0', 'RMS_RED', 'var_ED', 'var_RED', 'WCE', 'mue_ED', 'mue_ARED', 'single-sided'],
             r'$l_{mlp}$':['WCE', 'var_ED', 'NMED', 'var_ARED', 'mue_ARED', 'zero-error', 'mue_ED0', 'WCRE', 'mue_ED', 'RMS_ED', 'single-sided', 'var_RED', 'RMS_RED', 'mue_RED', 'ER', 'var_ED0'],
             'dfr':['mue_ED0', 'mue_ED', 'ER']
             }
    return res_c, res_r

def evaluationModelClassify(feature_sel, model, model_name, test_or_val=True):
    """
    模型的预测以及评分
    :param feature_sel: list  训练选择的特征
    :param model: 训练好的模型
    :param model_name: 模型名称
    :return: 模型的评价指标, list top1 top2 recall-1等
    """

    y, y_pre = predictClassify(model, feature_sel, model_name, test_or_val)
    y = np.array(y)
    result =evaluation(y, y_pre)

    return result

def classifyDraw(df, savename):
    """
    根据DataFrame绘制折线图
    :param df: DataFarme
    :param savename: 需要保存的名字
    :return:
    """
    plt.style.use(['science', 'ieee'])
    df = df.sort_values(by='macro-tpr', ascending=False)
    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    # plt.legend(label)
    plt.legend(loc='lower left')
    plt.savefig('../result/' + savename, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    res_c, res_r = sel_res()
    df = pd.read_csv('../../error/source/train_norm.csv')
    dt_df = pd.DataFrame(index=res_c.keys(), columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    svm_df = pd.DataFrame(index=res_c.keys(), columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    rf_df = pd.DataFrame(index=res_c.keys(), columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    mlp_df = pd.DataFrame(index=res_c.keys(), columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])

    df_dict = {
               'dt': dt_df,
               'svm': svm_df,
               'rf':rf_df,
               'mlp': mlp_df,
               }
    df = processData(df)

    # 添加模型
    func_dict = {}
    # func_dict['dt'] = dt.classifyDecisionTree
    # func_dict['svm'] = svm.classifySVM
    func_dict['rf'] = rf.classifyRF
    # func_dict['mlp'] = mlp.classifyMLP

    fixed_feature = ['net', 'dataset', 'concat']
    count = 0
    for key in res_c:
        print(count)
        count += 1
        feature_sel = res_c[key]

        for func in func_dict:
            model = func_dict[func](df, feature_sel + fixed_feature)
            acc = evaluationModelClassify(feature_sel + fixed_feature, model, func, True)
            df_dict[func].loc[key, :] = acc
    for key in func_dict:
        classifyDraw(df_dict[key], 'cla_'+key+'_sel.pdf')