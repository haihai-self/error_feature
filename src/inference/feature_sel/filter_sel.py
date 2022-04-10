from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, mutual_info_regression
from sklearn import metrics
import numpy as np
import pandas as pd
# import pymrmr
import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee'])





def varSel(df, ):
    sel_feature = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
            'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE']
    feature_var = []
    for index in sel_feature:
        feature_var.append(df.loc[:, index].var())
    var = pd.Series(index=sel_feature, data=feature_var)
    return var


def chi2Sel(df):
    sel_feature = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
            'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE']
    model = SelectKBest(chi2, k=14)
    data = df.loc[:, sel_feature].astype(float)
    target = df.loc[:, 'classify']

    target.astype(int)

    model.fit(data, target)
    score = -np.log(model.pvalues_)
    se = pd.Series(index= sel_feature,
                   data=score)
    return se


def regressionMrmrSel(df):
    sel_feature = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED',
                   'mue_ARED', 'var_ARED', 'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE']
    d = df.loc[:, sel_feature].astype(float)
    f = df.loc[:, 'untrained_acc']

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
    se = pd.Series(index=sel_feature, data=score).rename_axis(r'$C_{mrmr}$')
    # idxs = sorted(range(len(score)), key=lambda k: score[k], reverse=True)  # 得到与之对应的idx
    # score = sorted(score, reverse=True)  # 互信息排序
    return se


def classifyMrmrSel(df):
    sel_feature = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                   'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'single-sided', 'zero-error']
    continuous_feature = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                   'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE']
    discrete_feature = ['single-sided', 'zero-error']
    continuous_data = df.loc[:, continuous_feature].astype(float)
    discrete_data = df.loc[:, discrete_feature].astype(int)
    target = df.loc[:, 'classify']
    # 离散化连续变量，按照值
    d_cut = continuous_data.copy()
    for index, data in d_cut.iteritems():
        t = pd.cut(data, 25, duplicates='drop') #25类
        index_map = {label:idx for idx, label in enumerate(set(t))}
        t = pd.DataFrame(t)
        d_cut.loc[:, index] = t
        d_cut.loc[:, index] = d_cut.loc[:, index].map(index_map)


    # 离散化连续变量，按照值
    q_cut = continuous_data.copy()
    for index, data in q_cut.iteritems():
        t = pd.qcut(data, 50, duplicates='drop')
        index_map = {label:idx for idx, label in enumerate(set(t))} #36类

        t = pd.DataFrame(t)
        q_cut.loc[:, index] = t
        q_cut.loc[:, index] = q_cut.loc[:, index].map(index_map)

    d_cut = pd.merge(d_cut, discrete_data, left_index=True, right_index=True)
    q_cut = pd.merge(q_cut, discrete_data, left_index=True, right_index=True)

    # 计算d_cut
    # 计算与label的互信息
    mi_y = mutual_info_classif(d_cut, target)

    # 计算特征之间的互信息平均值
    mi_x = []
    for index, data in d_cut.iteritems():
        target = d_cut.loc[:, index]
        input = d_cut.drop(columns=index)
        temp = mutual_info_classif(input, target)
        mi_x.append(temp.sum() / input.shape[1])
    # 计算phi得分
    score = mi_y - mi_x
    se_d = pd.Series(index=sel_feature, data=score).rename_axis(r'$Dd_{mrmr}$')


    # 计算q_cut
    # 计算与label的互信息
    mi_y = mutual_info_classif(q_cut, target)

    # 计算特征之间的互信息平均值
    mi_x = []
    for index, data in q_cut.iteritems():
        target = q_cut.loc[:, index]
        input = q_cut.drop(columns=index)
        temp = mutual_info_classif(input, target)
        mi_x.append(temp.sum() / input.shape[1])
    # 计算phi得分
    score = mi_y - mi_x
    se_q = pd.Series(index=sel_feature, data=score).rename_axis(r'$Dq_{mrmr}$')


    return se_d, se_q


def cartPlot(df, loc, label_loc, label, tittle):
    df_plot = df.rename(columns={'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$', 'NMED': r'$NMED$',
                       'var_ED': r'$\sigma_{ED}$', 'mue_RED': r'$\mu_{RE}$', 'var_RED': r'$\sigma_{RE}$',
                       'mue_ARED': r'$\mu_{RED}$', 'var_ARED': r'$\sigma_{RED}$', 'RMS_ED': r'$rms_E$ ',
                       'RMS_RED': r'$rms_{RE}$', 'ER': r'$ER$', 'WCE': r'$W_E$', 'WCRE': r'$W_{RE}$', 'single-sided':r'$E_{ss}$', 'zero-error':r'$E_{zo}$'
                       })
    df_plot = df_plot.T
    df_plot.sort_values(by=label[1], inplace=True, ascending=False)
    plt.figure(figsize=(8, 4))
    for index, data in df_plot.iteritems():
        data = (data - data.min()) / (data.max() - data.min())
        plt.plot(df_plot.index, data, label=index)

    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.title(tittle)
    plt.yticks([x/10 for x in range(0, 12, 2)])
    plt.legend(loc=label_loc, fontsize=10)
    plt.savefig(loc, bbox_inches='tight')
    plt.show()
    plt.close()

    for index, data in df_plot.iteritems():
        df_plot.sort_values(by=index, inplace=True, ascending=False)
        print(df_plot.index)


if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df.loc[df.loc[:, 'single-sided'] == 'NO', 'single-sided'] = 0
    df.loc[df.loc[:, 'single-sided'] == 'YES', 'single-sided'] = 1
    df.loc[df.loc[:, 'zero-error'] == 'NO', 'zero-error'] = 0
    df.loc[df.loc[:, 'zero-error'] == 'YES', 'zero-error'] = 1

    var = varSel(df)
    chi2 = chi2Sel(df)
    index_c = [r'$\sigma$', r'$\chi^2$', r'$C_{mrmr}$']
    c_mrmr = re_mrmr = regressionMrmrSel(df)


    df_sel_d = pd.DataFrame([var,chi2, c_mrmr], index=index_c)
    saving_loc = '../result/filter_c.pdf'
    legend_loc = 'upper right'
    cartPlot(df_sel_d, saving_loc, legend_loc, label=index_c, tittle='continues feature importance')


    dd_mrmr, dq_mrmr = classifyMrmrSel(df)
    index_d = [r'$Dd_{mrmr}$', r'$Dq_{mrmr}$']
    df_sel_d = pd.DataFrame([dd_mrmr,dq_mrmr], index=index_d)
    saving_loc = '../result/filter_d.pdf'
    legend_loc = 'upper right'
    cartPlot(df_sel_d, saving_loc, legend_loc, label=index_d, tittle='discrete feature importance')

