import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def feature2latex():
    # dt_cla_fea = ['RMS_RED', 'WCRE', 'mue_ED']
    # str2latex(dt_cla_fea)
    # rf_cla_fea = ['var_ED0', 'var_RED', 'mue_ED0']
    # str2latex(rf_cla_fea)
    # svm_cla_fea = ['ER', 'mue_ED0', 'var_ED']
    # str2latex(svm_cla_fea)
    # mlp_cla_fea = ['var_ARED', 'ER', 'RMS_ED']
    # str2latex(mlp_cla_fea)

    # dt_reg_fea = ['mue_ED', 'RMS_RED', 'mue_ED0']
    # str2latex(dt_reg_fea)
    # rf_reg_fea = ['mue_ED0', 'var_ED0', 'mue_ARED']
    # str2latex(rf_reg_fea)
    svm_reg_fea = ['WCE', 'ER', 'mue_ARED']
    str2latex(svm_reg_fea)
    # mlp_reg_fea = ['var_RED', 'mue_ED', 'mue_ED0']
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
    data = [[97.40068661108387, 98.43060323688081, 95.11278195488721, 97.40068661108387, 88.45246893011378], # RMS_RED', 'WCRE', 'mue_ED'
            [98.52869053457577, 99.41147621383031, 97.36842105263158, 98.52869053457577, 89.53380599370298], # 'var_ED0', 'var_RED', 'mue_ED0'
            [96.17459538989701, 98.0872976949485, 93.23308270676691, 96.17459538989701,  73.41863324392595], # 'ER', 'mue_ED0', 'var_ED'
            [96.46885728298186, 99.16625796959295, 91.35338345864662, 96.46885728298186, 79.71030582159656]] # 'var_ARED', 'ER', 'RMS_ED'
    data = np.array(data)
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
    data = [[8.688135868367432, 93.3001955340236],  # 'mue_ED', 'RMS_RED', 'mue_ED0'
            [9.38124806487342, 94.87308612377005], #'mue_ED0', 'var_ED0', 'mue_ARED'
            [115.91031982908446, 76.07126594479092], # 'WCE', 'ER', 'mue_ARED'
            [10.552250338417672, 87.33803197048786]] # 'var_RED', 'mue_ED', 'mue_ED0'
    data = np.array(data)
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
    df_plot = df.copy()
    df_plot = df_plot.T
    df_plot.sort_values(by=0, ascending=False, inplace=True, axis=0)
    print(df_plot.index)
    df_plot = df_plot.T
    df_plot = df.rename(
        columns={'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$',
                 'var_ED': r'$\sigma_{ED}$', 'mue_RED': r'$\mu_{RE}$', 'var_RED': r'$\sigma_{RE}$',
                 'mue_ARED': r'$\mu_{RED}$', 'var_ARED': r'$\sigma_{RED}$', 'RMS_ED': r'$rms_E$ ',
                 'RMS_RED': r'$rms_{RE}$', 'ER': r'$ER$', 'WCE': r'$W_E$', 'WCRE': r'$W_{RE}$',
                 'single': r'$E_{ss}$', 'zero': r'$E_{zo}$'
                 })
    df_plot = df_plot.T
    df_plot.sort_values(by=0, ascending=False, inplace=True, axis=0)
    plt.figure(figsize=(9, 5))
    count = 0
    plt.style.use('science')

    for index, data in df_plot.iteritems():
        plt.plot(df_plot.index, data)
        count += 1
        if count == 10:
            break

    # plt.title('DropRank classify')
    plt.ylabel(r'keep rate ($\%$)')

    # plt.legend(loc='lower left')
    plt.savefig('result/drop_rank_cla.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def dropRankReg():
    # 绘制droprank得到的特征重要性排序图--回归模型
    df = pd.read_csv('result/regression_drop_rank.csv')
    df_plot = df.copy()
    df_plot = df_plot.T
    df_plot.sort_values(by=0, ascending=False, inplace=True, axis=0)
    print(df_plot.index)
    df_plot = df.rename(
        columns={'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$',
                 'var_ED': r'$\sigma_{ED}$', 'mue_RED': r'$\mu_{RE}$', 'var_RED': r'$\sigma_{RE}$',
                 'mue_ARED': r'$\mu_{RED}$', 'var_ARED': r'$\sigma_{RED}$', 'RMS_ED': r'$rms_E$ ',
                 'RMS_RED': r'$rms_{RE}$', 'ER': r'$ER$', 'WCE': r'$W_E$', 'WCRE': r'$W_{RE}$',
                 'single_side': r'$E_{ss}$', 'zero_error': r'$E_{zo}$'
                 })
    df_plot = df_plot.T
    df_plot.sort_values(by=0, ascending=False, inplace=True, axis=0)
    print(df_plot.index)
    plt.style.use('science')
    plt.figure(figsize=(9, 5))
    count = 0

    for index, data in df_plot.iteritems():
        plt.plot(df_plot.index, data)
        count += 1
        if count == 10:
            break

    # plt.title('DropRank classify')
    plt.ylabel(r'keep rate ($\%$)')

    # plt.legend(loc='lower left')
    plt.savefig('result/drop_rank_reg.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def treeRegModel():
    # 绘制树模型回归指标
    df_dt = pd.read_csv('inference/result/error_model/reg_dt_model.csv', index_col=0)
    df_dt = df_dt.rename(columns={'MAPE': r'$\text{MAPE}_{dt}$',
                                  r'$\chi^2$': r'$R^2_{dt}$'})
    df_rf = pd.read_csv('inference/result/error_model/reg_rf_model.csv', index_col=0)
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


def threshold(df, type='retrain'):
    df_plt = df[df['concat'] == 'vgg16cifar']

    temp = df_plt.loc[df_plt.loc[:, 'classify'] != 2, :]
    temp.to_csv('./result/vgg16_retrain.csv', index=False)

    plt.style.use(['science', 'ieee'])
    ax = plt.axes(projection='3d')
    feature_sel = ['mue_ED0', 'mue_ED', 'ER']

    marker = ['o', 'X', 'v', '^']
    edge_c = ['black', 'red', 'g', 'b']

    if type == 'retrain':
        class_list = [2, 1, 0, -1]
        pdf_name = 'threshold_3d_r.pdf'
        legend_loc = [.2, .8]
        ncol=2
    else:
        class_list = [2, 1, 0]
        pdf_name = 'threshold_3d_i.pdf'
        legend_loc = [.2, .8]
        ncol=2

    for i in class_list:
        x = df_plt.loc[(df.loc[:, 'classify'] == i) & ((df.loc[:, 'mue_ED0']) < 1024), feature_sel[0]]

        y = df_plt.loc[x.index, feature_sel[1]]
        x = np.log(x + 1)
        y = np.log(y + 1)

        z = df_plt.loc[x.index, feature_sel[2]]

        if i != -1:
            ax.scatter3D(x, y, z, label='class %d' % (i), alpha=0.9, edgecolors=edge_c[i], marker=marker[i], c='w')
        else:
            ax.scatter3D(x, y, z, label='class b', alpha=0.9, edgecolors=edge_c[i], marker=marker[i], c='w')
    ax.set_xlabel(r'$\log_2(\mu_E)$')
    ax.set_ylabel(r'$\log_2(\mu_{ED})$')
    ax.set_zlabel(r'$ER$')

    ax.view_init(10, 150)

    plt.legend(loc=legend_loc, ncol=ncol)
    plt.savefig('result/' + pdf_name)

    # plt.legend(loc=[.55, .1])
    # plt.savefig('result/threshold_3d_i.pdf')

    # plt.show()
    plt.close()


def threshold2d(df, type='retrain'):
    df_plt = df[df['concat'] == 'vgg16cifar']

    plt.style.use(['science', 'ieee'])
    feature_sel = ['mue_ED0', 'mue_ED', 'ER']

    marker = ['o', 'X', 'v', '^']
    edge_c = ['black', 'red', 'g', 'b']
    fig, ax = plt.subplots(1, 1)
    if type == 'retrain':
        class_list = [2, 1, 0, -1]
        pdf_name = 'threshold_2d_r.pdf'
        axins = ax.inset_axes((.45, .1, .25, .25))
        text_loc = [0.5, 0.03]
        class_range=[25, 31]
    else:
        class_list = [2, 1, 0]
        pdf_name = 'threshold_2d_i.pdf'
        axins = ax.inset_axes((.44, .1, .25, .25))
        text_loc = [0.4, 0.08]
        class_range=[8, 10.5]


    for i in class_list:
        x = df_plt.loc[(df.loc[:, 'classify'] == i) & ((df.loc[:, 'mue_ED0']) < 1024), feature_sel[0]]
        y = df_plt.loc[(x.index), feature_sel[1]]
        x = np.log2(x + 1)
        y = np.log2(y + 1)

        if i != -1:
            ax.scatter(x, y, label='class %d' % (i), alpha=0.9, edgecolors=edge_c[i], marker=marker[i], c='w', lw=1,
                       s=20)
        else:
            ax.scatter(x, y, label='class b', alpha=0.9, edgecolors=edge_c[i], marker=marker[i], c='w', lw=1, s=20)

    plt.ylabel(r'$\log_2(\mu_{ED})$')
    plt.xlabel(r'$\log_2(\mu_E)$')
    plt.legend(loc='lower right')

    class_list.reverse()
    for i in class_list:
        y = df_plt.loc[
            (df.loc[:, 'classify'] == i) & ((df.loc[:, 'mue_ED'] < class_range[1]) & (df.loc[:, 'mue_ED'] > class_range[0])), feature_sel[1]]
        x = df_plt.loc[(y.index), feature_sel[0]]
        x = np.log2(x)
        y = np.log2(y)
        axins.scatter(x, y, alpha=0.8, edgecolors=edge_c[i], marker=marker[i], c='w', s=20)
        if i == 2:
            axins.text(x - text_loc[0], y + text_loc[1],
                       s='(' + '%.1f' % x.values[0] + ',' + '%.1f' % y.values[0] + ')', size=7)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1, linestyle='--')

    plt.savefig('result/' + pdf_name)

    plt.close()


def cla_feature_sel():
    """

    :return:
    """
    names = []
    names.append('cla_mlp_sel')
    names.append('cla_dt_sel')
    names.append('cla_rf_sel')
    names.append('cla_svm_sel')
    for name in names:
        file_path = 'result/cla_evaluation_feature_sel/' + name + '.csv'
        save_path = 'result/cla_evaluation_feature_sel/' + name + '.pdf'
        plt.style.use(['science', 'ieee'])
        df = pd.read_csv(file_path, index_col=0)
        for index, data in df.iteritems():
            if index == 'weight-tpr':
                continue
            plt.plot(df.index, data.values, label=index)
        plt.legend(loc='lower left')
        plt.xlabel('Feature selection method')
        plt.ylabel(r'accuracy ($\%$)')
        plt.savefig(save_path)
        plt.close()

def reg_feature_sel():
    names = []
    names.append('reg_mlp_sel')
    names.append('reg_dt_sel')
    names.append('reg_rf_sel')
    names.append('reg_svm_sel')
    for name in names:
        file_path = 'result/reg_evaluation_feature_sel/' + name + '.csv'
        save_path = 'result/reg_evaluation_feature_sel/' + name + '.pdf'
        plt.style.use(['science', 'ieee'])
        df = pd.read_csv(file_path, index_col=0)
        for index, data in df.iteritems():
            plt.plot(df.index, data.values, label=index)
        plt.xlabel('Feature selection method')
        plt.ylabel(r'accuracy ($\%$)')
        plt.legend(loc='lower left')
        plt.savefig(save_path)
        plt.close()

def cla_model_mlp_retrain():
    df_lists = []
    feature_len = 17
    data_path = 'result/feature_sel_res/'
    for i in range(1, feature_len):
        df = pd.read_csv(data_path + 'cla_mlp_model' + str(i) + '.csv')
        df_lists.append(df)
    graph_names = ['top-1', 'top-2', 'recall-1', 'macro-tpr', 'weight-tpr']
    y_labels = ['top-1', 'top-2', 'recall-1', 'macro-tpr', 'weight-tpr']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar','resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    line = ['-', '--', ':', '-.', '--', ':', '-.', '--']
    color = ['k', 'b', 'b', 'b', 'g', 'g', 'g', 'r']
    for j in range(len(graph_names)):
        name = graph_names[j]
        df_plot = pd.DataFrame(index=indexes)
        for i in range(len(df_lists)):
            insert_col = df_lists[i].loc[:, name]
            df_plot[str(i+1)] = insert_col.values
        plt.style.use(['science', 'ieee'])
        count = 0
        for index, row in df_plot.iterrows():
            plt.plot(row.index, row, color=color[count], linestyle=line[count])
            count+=1

        plt.xlabel('number of error features')
        plt.ylabel(y_labels[j] + r' ($\%$)')
        plt.savefig(data_path + name + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

def reg_model_mlp_retrain():
    df_lists = []
    feature_len = 17
    data_path = 'result/feature_sel_res/'
    for i in range(1, feature_len):
        df = pd.read_csv(data_path + 'reg_mlp_model' + str(i) + '.csv')
        df_lists.append(df)
    graph_names = ['MAPE', r'$\chi^2$']
    y_label = ['MAPE', r'$R^2$']
    save_name = ['MAPE', 'R2']
    indexes = ['domain', 'vgg16-mnist', 'resnet18-mnist', 'resnet34-mnist', 'vgg16-cifar10','resnet18-cifar10',
               'resnet34-cifar10', 'resnet34-cifar100']
    line = ['-', '--', ':', '-.', '--', ':', '-.', '--']
    color = ['k', 'b', 'b', 'b', 'g', 'g', 'g', 'r']
    for j in range(len(graph_names)):
        name = graph_names[j]
        df_plot = pd.DataFrame(index=indexes)
        for i in range(len(df_lists)):
            insert_col = df_lists[i].loc[:, name]
            df_plot[str(i+1)] = insert_col.values
        plt.style.use(['science', 'ieee'])
        count = 0
        for index, row in df_plot.iterrows():
            plt.plot(row.index, row, color=color[count], linestyle=line[count], label=index)
            count+=1
        # plt.legend(loc='best')
        plt.xlabel('number of error features')
        plt.ylabel(y_label[j] + r' ($\%$)')
        plt.savefig(data_path + save_name[j] + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()


def cla_error_model():
    model_names = ['dt', 'rf', 'mlp', 'svm']
    df_lists = []
    for name in model_names:
        df = pd.read_csv('./result/error_model/cla_' + name + '_model.csv', index_col=0)
        df_lists.append(df)
    graph_names = ['top-1', 'top-2', 'recall-1', 'macro-tpr', 'weight-tpr']
    y_labels = ['top-1', 'top-2', 'recall-1', 'macro-tpr', 'weight-tpr']
    for i in range(len(graph_names)):
        name = graph_names[i]
        df_plot = pd.DataFrame(index=df_lists[0].index)
        for j in range(len(df_lists)):
            insert_col = df_lists[j].loc[:, name]
            df_plot[model_names[j]] = insert_col.values
        plt.style.use(['science', 'ieee'])
        for index, data in df_plot.iteritems():
            plt.plot(df_plot.index, data.values, label=index)
        # plt.legend(loc='best')
        plt.xticks(rotation=300)
        plt.ylabel(y_labels[i] + r' ($\%$)')
        plt.savefig('./result/error_model/cla_' + name + '_model.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
def reg_error_model():
    model_names = ['dt', 'rf', 'mlp']
    df_lists = []
    for name in model_names:
        df = pd.read_csv('./result/error_model/reg_' + name + '_model.csv', index_col=0)
        df_lists.append(df)
    y_labels = ['MAPE', r'$R^2$']
    col_name = ['MAPE', r'$\chi^2$']
    graph_names = ['MAPE', 'R2']
    for i in range(len(graph_names)):
        name = col_name[i]

        df_plot = pd.DataFrame(index=df_lists[0].index)
        for j in range(len(df_lists)):
            insert_col = df_lists[j].loc[:, name]
            df_plot[model_names[j]] = insert_col.values
        plt.style.use(['science', 'ieee'])
        for index, data in df_plot.iteritems():
            plt.plot(df_plot.index, data.values, label=index)
        # plt.legend(loc='best')
        plt.xticks(rotation=300)
        plt.ylabel(y_labels[i] + r' ($\%$)')
        plt.savefig('./result/error_model/reg_' + graph_names[i] + '_model.pdf', bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == '__main__':
    # cla_feature_sel()
    reg_feature_sel()
    # cla_mlp_retrain()
    # reg_mlp_zero_out()
    # cla_model_mlp_retrain()
    # reg_model_mlp_retrain()
    # cla_error_model()
    # reg_error_model()
    # feature2latex()
    # dropRankCla()
    # dropRankReg()
    # wrapperClassify()
    # wrapperRegression()
    # treeRegModel()
    # df_r = pd.read_csv('../error/source/retrain_dataset.csv')
    # df_i = pd.read_csv('../error/source/dataset.csv')
    #
    # threshold2d(df_i, 'unretrain')
    # threshold(df_i, 'unretrain')
    # threshold2d(df_r, 'retrain')
    # threshold(df_r, 'retrain')
