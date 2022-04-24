import pandas as pd
import matplotlib.pyplot as plt
import os


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

def normDataset():
    df = pd.read_csv("source/err.csv", index_col='mul_name')
    df_acc = pd.read_csv('source/untrained_acc.csv', index_col='mul_name')
    df = pd.merge(df, df_acc, on='mul_name')

    # 分割训练集测试集

    # df = df.sample(frac=1.0, random_state=10)  # 全部打乱
    # cut_idx = int(round(0.25 * df.shape[0]))
    # df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    df_train = pd.read_csv('./source/train_data.csv')
    df_test = pd.read_csv('./source/test_data.csv')
    df_val = pd.read_csv('./source/val_data.csv')
    df_dataset = pd.concat([df_train, df_test])
    df_dataset.to_csv('./source/dataset.csv', index=False)

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
        df_train_not_max.loc[:, index] = (df_train_not_max.loc[:, index] - des.loc['min', index]) / (
                    des.loc['max', index] - des.loc['min', index])

        # 设置测试集
        df_test.loc[df_test.loc[:, index] >= des.loc['73%', index], index] = des.loc['73%', index]
        df_test.loc[:, index] = (df_test.loc[:, index] - des.loc['min', index]) / (
                des.loc['73%', index] - des.loc['min', index])

        # 设置验证集
        df_val.loc[df_val.loc[:, index] >= des.loc['73%', index], index] = des.loc['73%', index]
        df_val.loc[:, index] = (df_val.loc[:, index] - des.loc['min', index]) / (
                des.loc['73%', index] - des.loc['min', index])

    # 保存
    df_train_set_max.to_csv('./source/train_norm.csv', index=False)
    df_test.to_csv('./source/test_norm.csv', index=False)
    df_val.to_csv('./source/val_norm.csv', index=False)

    # densPlot(df_train_set_max, './result/density_setmax.pdf', 'upper left')
    # densPlot(df_train_not_max, './result/density_nomax.pdf', 'upper right')

def getRetrain():
    target_dir = ['vgg16_cifar10_origin_app', 'vgg16_mnist_origin_app']
    retrain_path  = '../../result/retrain'
    df = pd.DataFrame(columns=['mul_name', 'trained_acc', 'net', 'dataset', 'concat'])

    # 读取应用文件夹
    for app in target_dir:
        app_path = retrain_path + '/' + app
        temp = app.strip().split('_')
        net_name = temp[0]
        dataset_name = temp[1]
        concat_name = net_name+dataset_name

        method_dirs = os.listdir(app_path)

        # 读取近似计算方法文件夹
        for method_dir in method_dirs:
            method_path = app_path + '/' + method_dir
            mul_dirs = os.listdir(method_path)
            # 读取乘法器文件夹
            for mul_dir in mul_dirs:
                mul_path = method_path + '/' + mul_dir
                acc_path = mul_path + '/acc.log'
                mul_name = mul_dir.split('.')[0]
                # 读取每个AM的准确率
                with open(acc_path, 'r') as f:
                    acc = f.readline().strip().split(' ')[3]
                    new_col = {'mul_name':mul_name, 'trained_acc':acc, 'net':net_name, 'dataset':dataset_name, 'concat':concat_name}
                    new_col = new_col.values()
                    row_len = df.shape[0]
                    # 添加新数据到df
                    df.loc[row_len] = new_col
    # 保存csv
    df.to_csv('source/retrain.csv', index=False)





if __name__ == '__main__':
    getRetrain()

