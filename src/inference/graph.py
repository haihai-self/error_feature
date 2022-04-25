import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def feature2latex():
    treeRegModel()
    # dt_cla_fea = ['var_RED', 'single-sided', 'var_ARED', 'mue_ED0', 'ER', 'mue_ARED', 'mue_RED', 'var_ED', 'RMS_ED', 'NMED', 'WCE']
    # str2latex(dt_cla_fea)
    # rf_cla_fea = ['var_ED0', 'WCE', 'mue_ED0', 'var_RED', 'RMS_RED']
    # str2latex(rf_cla_fea)
    # svm_cla_fea = ['var_ED', 'ER', 'WCE', 'WCRE', 'mue_RED', 'RMS_ED', 'RMS_RED', 'mue_ARED', 'single-sided', 'mue_ED', 'var_ARED', 'mue_ED0', 'var_RED']
    # str2latex(svm_cla_fea)
    # mlp_cla_fea = ['NMED', 'RMS_RED', 'mue_ED0', 'WCE', 'mue_ARED', 'var_ED0', 'RMS_ED', 'zero-error', 'ER', 'mue_RED', 'mue_ED', 'single-sided']
    # str2latex(mlp_cla_fea)

    # dt_reg_fea = ['WCE', 'mue_ARED', 'mue_ED', 'NMED', 'ER', 'mue_ED0', 'mue_RED', 'RMS_ED', 'WCRE', 'var_RED', 'var_ARED', 'var_ED0', 'zero-error']
    # str2latex(dt_reg_fea)
    # rf_reg_fea = ['var_ARED', 'RMS_ED', 'zero-error', 'mue_ED0', 'WCRE', 'var_ED0', 'RMS_RED', 'var_ED', 'var_RED', 'WCE', 'mue_ED', 'mue_ARED', 'single-sided']
    # str2latex(rf_reg_fea)
    # svm_reg_fea = ['WCE', 'var_ED', 'NMED', 'var_ARED', 'mue_ARED', 'zero-error', 'mue_ED0', 'WCRE', 'mue_ED', 'RMS_ED', 'single-sided', 'var_RED', 'RMS_RED', 'mue_RED', 'ER', 'var_ED0']
    # str2latex(svm_reg_fea)
    # mlp_reg_fea = ['ER', 'mue_ED0', 'zero-error', 'NMED', 'mue_ED', 'var_ARED', 'single-sided', 'var_RED', 'WCRE', 'RMS_RED', 'WCE', 'mue_ARED', 'RMS_ED', 'var_ED']
    # str2latex(mlp_reg_fea)


def str2latex(list):
    # 将str名称修改成latex格式打印出来
    latex_dict = {'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$', 'NMED': r'$NMED$',
                  'var_ED': r'$\sigma_{ED}$', 'mue_RED': r'$\mu_{RE}$', 'var_RED': r'$\sigma_{RE}$',
                  'mue_ARED': r'$\mu_{RED}$', 'var_ARED': r'$\sigma_{RED}$', 'RMS_ED': r'$rms_E$ ',
                  'RMS_RED': r'$rms_{RE}$', 'ER': r'$ER$', 'WCE': r'$W_E$', 'WCRE': r'$W_{RE}$',
                  'single-sided': r'$E_{ss}$', 'zero-error': r'$E_{zo}$'
                  }
    for i in list:
        print(latex_dict[i], end=', ')
    print()


def wrapperClassify():
    # 得到lvm方法分类模型对应的指标
    data = [[0.9759686120647376, 0.9877390877881315, 0.9360902255639098, 0.9759686120647376, 0.8871580552962045],
            [0.9857773418342325, 0.9926434526728789, 0.9736842105263158, 0.9857773418342325, 0.8894363486601541],
            [0.9749877390877881, 0.9906817067189799, 0.9586466165413534, 0.9749877390877881, 0.8504578966063688],
            [0.973516429622364, 0.9916625796959294, 0.9548872180451128, 0.973516429622364, 0.8853902252339932]]
    data = np.array(data)
    data = data * 100
    columns = ['acc-1', 'acc-2', 'recall-1', 'weight-tpr', 'macro-tpr']
    index = ['DT', 'RF', 'SVM', 'MLP']
    plt.style.use(['science', 'ieee'])

    df = pd.DataFrame(data=data, columns=columns, index=index).T

    # plt.figure(figsize=(8, 4))
    for index, data in df.iteritems():
        plt.plot(df.index, data, label=index)

    # plt.title('lvm classify feature select')
    # plt.yticks([x / 10 for x in range(8, 11)])
    plt.legend(loc='lower left')
    plt.savefig('result/lvm_sel.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def wrapperRegression():
    # 得到lvm方法回归模型对应的指
    data = [[0.07773549004573779, 0.9468117451601555],
            [0.08881568010298237, 0.9611718110937534],
            [0.2545986885971268, 0.8520471294395444],
            [0.06844977317607073, 0.9235117926731133]]
    data = np.array(data)
    data = data * 100
    columns = ['mape', r'$R^2$']
    index = ['DT', 'RF', 'SVM', 'MLP']
    plt.style.use(['science', 'ieee'])

    df = pd.DataFrame(data=data, columns=columns, index=index).T

    # plt.figure(figsize=(8, 4))
    plt.style.use(['science'])
    size = 2
    x = np.arange(size)
    total_width, n = 0.8, 4
    width = total_width / n
    x = x - (total_width - width) / 2

    count = 0
    for index, data in df.iteritems():
        tick_label = None
        if count == 2:
            tick_label = columns
        plt.bar(x + count * width, data, width=width, label=index, tick_label=tick_label, align='edge')
        count += 1
        # plt.hist(df.index, data, label=index)

    # plt.title('lvm classify feature select')
    plt.xticks(label=['mape', r'$R^2$'])
    plt.legend(loc='upper left')
    plt.savefig('result/lvm_regression.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def dropRankCla():
    # 绘制droprank得到的特征重要性排序图--分类模型
    df = pd.read_csv('result/drop.csv')
    df_plot = df.rename(
        columns={'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$',
                 'var_ED': r'$\sigma_{ED}$', 'mue_RED': r'$\mu_{RE}$', 'var_RED': r'$\sigma_{RE}$',
                 'mue_ARED': r'$\mu_{RED}$', 'var_ARED': r'$\sigma_{RED}$', 'RMS_ED': r'$rms_E$ ',
                 'RMS_RED': r'$rms_{RE}$', 'ER': r'$ER$', 'WCE': r'$W_E$', 'WCRE': r'$W_{RE}$',
                 'single': r'$E_{ss}$', 'zero': r'$E_{zo}$'
                 })
    df_plot = df_plot.T
    plt.figure(figsize=(9, 5))
    count = 0
    plt.style.use('science')
    df_plot.sort_values(by=0, ascending=False, inplace=True, axis=0)
    print(df_plot.index)
    for index, data in df_plot.iteritems():
        plt.plot(df_plot.index, data)
        count += 1
        if count == 10:
            break

    plt.title('DropRank classify')
    # plt.legend(loc='lower left')
    plt.savefig('result/drop_rank_cla.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def dropRankReg():
    # 绘制droprank得到的特征重要性排序图--回归模型
    df = pd.read_csv('result/regression_drop_rank.csv')
    df_plot = df.rename(
        columns={'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$',
                 'var_ED': r'$\sigma_{ED}$', 'mue_RED': r'$\mu_{RE}$', 'var_RED': r'$\sigma_{RE}$',
                 'mue_ARED': r'$\mu_{RED}$', 'var_ARED': r'$\sigma_{RED}$', 'RMS_ED': r'$rms_E$ ',
                 'RMS_RED': r'$rms_{RE}$', 'ER': r'$ER$', 'WCE': r'$W_E$', 'WCRE': r'$W_{RE}$',
                 'single_side': r'$E_{ss}$', 'zero_error': r'$E_{zo}$'
                 })
    df_plot = df_plot.T
    plt.style.use('science')
    plt.figure(figsize=(9, 5))
    count = 0
    df_plot.sort_values(by=0, ascending=False, inplace=True, axis=0)
    print(df_plot.index)
    for index, data in df_plot.iteritems():
        plt.plot(df_plot.index, data)
        count += 1
        if count == 10:
            break

    plt.title('DropRank classify')
    # plt.legend(loc='lower left')
    plt.savefig('result/drop_rank_reg.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def treeRegModel():
    # 绘制树模型回归指标
    df_dt = pd.read_csv('result/csv/reg_dt_model.csv', index_col=0)
    df_dt = df_dt.rename(columns={'MAPE': r'$\text{MAPE}_{dt}$',
                                  r'$\chi^2$': r'$R^2_{dt}$'})
    df_rf = pd.read_csv('result/csv/reg_rf_model.csv', index_col=0)
    df_rf = df_rf.rename(columns={'MAPE': r'$\text{MAPE}_{rf}$',
                                  r'$\chi^2$': r'$R^2_{rf}$'})
    plt.style.use(['science', 'ieee'])
    df = pd.merge(df_dt, df_rf, left_index=True, right_index=True)
    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    # plt.legend(label)
    plt.legend(loc='best')
    plt.xticks(rotation=300)
    plt.savefig('result/tree_reg_model.pdf', bbox_inches='tight')
    plt.show()


def threshold(df):
    df_plt = df[df['concat'] == 'vgg16cifar']
    df_plt.sort_values(by='mue_ED0')

    plt.style.use(['science', 'ieee'])
    ax = plt.axes(projection='3d')
    feature_sel = ['mue_ED0', 'mue_ED', 'ER']

    marker = ['o', 'X', 'v']
    edge_c = ['black', 'red', 'g']

    for i in [2, 1, 0]:
        x = df_plt.loc[df.loc[:, 'classify'] == i, feature_sel[0]]
        y = df_plt.loc[df.loc[:, 'classify'] == i, feature_sel[1]]

        z = df_plt.loc[df.loc[:, 'classify'] == i, feature_sel[2]]

        ax.scatter3D(x, y, z, label='class%d' % (i), alpha=0.9, edgecolors=edge_c[i], marker=marker[i], c='w')
    ax.set_ylabel(r'$\mu_{ED}$')
    ax.set_xlabel(r'$\mu_E$')
    ax.set_zlabel(r'$ER$')

    plt.legend(loc=[.4, .2])
    plt.savefig('result/threshold_3d_i.pdf')
    plt.show()
    plt.close()


def threshold2d(df):
    df_plt = df[df['concat'] == 'vgg16cifar']
    # df_plt.sort_values(by='mue_ED', inplace=True)

    plt.style.use(['science', 'ieee'])
    feature_sel = ['mue_ED0', 'mue_ED', 'ER']

    marker = ['o', '^', 'v']
    edge_c = ['black', 'red', 'g']
    fig, ax = plt.subplots(1, 1)
    for i in [2, 1, 0]:
        x = df_plt.loc[(df.loc[:, 'classify'] == i) & ((df.loc[:, 'mue_ED0']) < 150), feature_sel[0]]
        y = df_plt.loc[(x.index), feature_sel[1]]

        # z = df_plt.loc[df.loc[:, 'classify'] == i, feature_sel[2]]

        ax.scatter(x, y, label='class%d' % (i), alpha=0.9, edgecolors=edge_c[i], marker=marker[i], c='w', lw=1)
    plt.ylabel(r'$\mu_{ED}$')
    plt.xlabel(r'$\mu_E$')
    plt.legend()
    axins = ax.inset_axes((.1, .65, .3, .3))
    #
    # y_list = []
    for i in [0, 1, 2]:
        y = df_plt.loc[(df.loc[:, 'classify'] == i) & ((df.loc[:, 'mue_ED']) < 10.5), feature_sel[1]]
        x = df_plt.loc[(y.index), feature_sel[0]]
        # z = df_plt.loc[df.loc[:, 'classify'] == i, feature_sel[2]]

        axins.scatter(x, y, alpha=0.9, edgecolors=edge_c[i], marker=marker[i], c='w')
        if i == 2:
            axins.text(x - 3, y - 5, s='(' + '%.1f' % x.values[0] + ',' + '%.1f' % y.values[0] + ')', size=7)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1, linestyle='--')
    plt.savefig('result/threshold_2d_i.pdf')
    plt.close()


if __name__ == '__main__':
    # dropRankCla()
    # dropRankReg()
    # wrapperClassify()
    # wrapperRegression()
    # treeRegModel()
    # df = pd.read_csv('../error/source/retrain_dataset.csv')
    df = pd.read_csv('../error/source/dataset.csv')

    # threshold2d(df)
    threshold(df)
