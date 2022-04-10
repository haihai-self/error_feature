import pandas as pd
import matplotlib.pyplot as plt


def densPlot(df, loc, label_loc):
    df = df.drop(columns=['untrained_acc', 'classify'])
    df_plot = df.rename(columns={'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$', 'NMED': r'$NMED$',
                       'var_ED': r'$\sigma_{ED}$', 'mue_RED': r'$\mu_{RE}$', 'var_RED': r'$\sigma_{RE}$',
                       'mue_ARED': r'$\mu_{RED}$', 'var_ARED': r'$\sigma_{RED}$', 'RMS_ED': r'$rms_E$ ',
                       'RMS_RED': r'$rms_{RE}$', 'ER': r'$ER$', 'WCE': r'$W_E$', 'WCRE': r'$W_{RE}$'
                       })
    # 绘制密度图像
    ax = df_plot.plot.kde()
    ax.legend(loc=label_loc, ncol=3, fontsize=10)
    fig = ax.get_figure()
    fig.show()
    fig.savefig(loc, bbox_inches='tight')

if __name__ == '__main__':
    df = pd.read_csv("source/err.csv", index_col='mul_name')
    df_acc = pd.read_csv('source/untrained_acc.csv', index_col='mul_name')
    df = pd.merge(df, df_acc, on='mul_name')

    # 分割训练集测试集
    df = df.sample(frac=1.0)  # 全部打乱
    cut_idx = int(round(0.25 * df.shape[0]))
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    df_train = pd.read_csv('./source/train_data.csv')
    df_test = pd.read_csv('./source/test_data.csv')

    # 得到训练集参数 用于归一化
    des = df_train.describe(percentiles=[0.5, 0.6, 0.7, 0.71, 0.72, 0.73, 0.74])
    df_train_set_max = df_train.copy()
    df_train_not_max = df_train.copy()
    for index, data in df.iteritems():
        if index == 'single-sided' or index == 'zero-error' or index == 'untrained':
            continue
        # 设置训练集
        df_train_set_max.loc[df_train_set_max.loc[:, index] >= des.loc['73%', index], index] = des.loc['73%', index]
        df_train_set_max.loc[:, index] = (df_train_set_max.loc[:, index] - des.loc['min', index]) / (
                des.loc['73%', index] - des.loc['min', index])

        # 不设置最大值归一化
        df_train_not_max.loc[:, index] = (df_train_not_max.loc[:, index] - des.loc['min', index]) / (des.loc['max', index] - des.loc['min', index])

        # 设置测试集
        df_test.loc[df_test.loc[:, index] >= des.loc['73%', index], index] = des.loc['73%', index]
        df_test.loc[:, index] = (df_test.loc[:, index] - des.loc['min', index]) / (
                des.loc['73%', index] - des.loc['min', index])

    # 保存
    df_train_set_max.to_csv('./source/train_norm.csv')
    df_test.to_csv('./source/test_norm.csv')
    densPlot(df_train_set_max, './result/density_setmax.pdf', 'upper left')
    densPlot(df_train_not_max, './result/density_nomax.pdf', 'upper right')

