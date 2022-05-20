import pandas as pd
import matplotlib.pyplot as plt
import os


def densPlot(df, loc, label_loc):
    df = df.drop(columns=['untrained_acc', 'classify'])
    df_plot = df.rename(
        columns={'mue_ED0': r'$\mu_E$', 'var_ED0': r'$\sigma_E$', 'mue_ED': r'$\mu_{ED}$', 'NMED': r'$NMED$',
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
    # df = pd.read_csv("source/err.csv", index_coldex_col='mul_name')
    # df_acc = pd.read_csv('source/untrained_acc.csv', index_col='mul_name')
    # df = pd.merge(df, df_acc, on='mul_name')

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
    for index, data in des.iteritems():
        if index == 'single-sided' or index == 'zero-error' or index == 'untrained_acc' or index == 'classify':
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
    """
    遍历retrain的文件夹，得到对应retrain之后的acc
    :return:
    """
    target_dir = ['vgg16_cifar10_origin_app', 'vgg16_mnist_origin_app']
    retrain_path = '../../result/retrain'
    df = pd.DataFrame(columns=['mul_name', 'retrained_acc', 'net', 'dataset', 'concat'])

    # 读取应用文件夹
    for app in target_dir:
        app_path = retrain_path + '/' + app
        temp = app.strip().split('_')
        net_name = temp[0]
        dataset_name = temp[1]
        concat_name = net_name + dataset_name

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
                    new_col = {'mul_name': mul_name, 'trained_acc': acc, 'net': net_name, 'dataset': dataset_name,
                               'concat': concat_name}
                    new_col = new_col.values()
                    row_len = df.shape[0]
                    # 添加新数据到df
                    df.loc[row_len] = new_col
    # 保存csv
    df.to_csv('source/retrain.csv', index=False)


def addRetrain2data():
    """
    将retrain的结果加入数据集
    :return:
    """
    df_retrain = pd.read_csv('source/retrain.csv')
    df_train = pd.read_csv('source/train_data.csv')
    df_test = pd.read_csv('source/test_data.csv')
    df_val = pd.read_csv('source/val_data.csv')
    df_train.insert(loc=1, column='retrained_acc', value=-1)
    df_test.insert(loc=1, column='retrained_acc', value=-1)
    df_val.insert(loc=1, column='retrained_acc', value=-1)

    for index, row in df_retrain.iterrows():
        if row['concat'] == 'vgg16cifar10':
            row['concat'] = 'vgg16cifar'
        df_train.loc[(df_train.loc[:, 'mul_name'] == row['mul_name']) & (
                df_train.loc[:, 'concat'] == row['concat']), 'retrained_acc'] = row['retrained_acc']
        df_test.loc[(df_test.loc[:, 'mul_name'] == row['mul_name']) & (
                df_test.loc[:, 'concat'] == row['concat']), 'retrained_acc'] = row['retrained_acc']
        df_val.loc[(df_val.loc[:, 'mul_name'] == row['mul_name']) & (
                df_val.loc[:, 'concat'] == row['concat']), 'retrained_acc'] = row['retrained_acc']

        # a = df_train.loc[:, 'mul_name'] row['mul_name']
        # b = df_retrain['concat'] == row['concat']
        # a = df_train.loc[df_train[:, 'mul_name'] == row['mul_name'] and df_retrain[:, 'concat'] == row['concat']]
    df_train.to_csv('./source/re_train_data.csv', index=False)
    df_test.to_csv('./source/re_test_data.csv', index=False)
    df_val.to_csv('./source/re_val_data.csv', index=False)
    val_vgg = df_val.loc[df_val.loc[:, 'concat'] == 'vgg16cifar', :]
    val_vgg.to_csv('./source/val_vgg.csv', index=False)


def normRetrain():
    """
    将retrain数据集归一化
    :return:
    """
    df_train = pd.read_csv('./source/re_train_data.csv')
    df_test = pd.read_csv('./source/re_test_data.csv')

    # 设置分类
    app_dict = {
        'vgg16cifar': 0.78521999764442444
    }
    for key, val in app_dict.items():
        # 将train数据分类
        df_train.loc[(df_train.loc[:, 'concat'] == key) & (df_train.loc[:, 'retrained_acc'] >= val), 'classify'] = -1
        df_train.loc[(df_train.loc[:, 'concat'] == key) & ((val - df_train.loc[:, 'retrained_acc']) <= 0.03) & (
                0 < (val - df_train.loc[:, 'retrained_acc'])), 'classify'] = 0
        df_train.loc[(df_train.loc[:, 'concat'] == key) & (
            (0.03 <val - df_train.loc[:, 'retrained_acc']) & (val - df_train.loc[:, 'retrained_acc'] <=0.08)), 'classify'] = 1
        df_train.loc[(df_train.loc[:, 'concat'] == key) & (
                (0.08 < val - df_train.loc[:, 'retrained_acc'])), 'classify'] = 2

        # 将test数据分类
        df_test.loc[(df_test.loc[:, 'concat'] == key) & (df_test.loc[:, 'retrained_acc'] >= val), 'classify'] = -1
        df_test.loc[(df_test.loc[:, 'concat'] == key) & ((val - df_test.loc[:, 'retrained_acc']) <= 0.03) & (
                0 < (val - df_test.loc[:, 'retrained_acc'])), 'classify'] = 0
        df_test.loc[(df_test.loc[:, 'concat'] == key) & (
                (0.03 < val - df_test.loc[:, 'retrained_acc']) & (
                    val - df_test.loc[:, 'retrained_acc'] <= 0.08)), 'classify'] = 1
        df_test.loc[(df_test.loc[:, 'concat'] == key) & (
            (0.08 < val - df_test.loc[:, 'retrained_acc'])), 'classify'] = 2

    df_dataset = pd.concat([df_train, df_test])
    df_dataset.to_csv('./source/retrain_dataset.csv', index=False)

    # 得到训练集参数 用于归一化
    des = df_train.describe(percentiles=[0.5, 0.6, 0.7, 0.71, 0.72, 0.73, 0.74])
    df_train_set_max = df_train.copy()
    df_train_not_max = df_train.copy()
    for index, data in des.iteritems():
        if index == 'single-sided' or index == 'zero-error' or index == 'untrained_acc' or index == 'retrained_acc' or index == 'classify':
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
        # df_val.loc[df_val.loc[:, index] >= des.loc['73%', index], index] = des.loc['73%', index]
        # df_val.loc[:, index] = (df_val.loc[:, index] - des.loc['min', index]) / (
        #         des.loc['73%', index] - des.loc['min', index])

    # 保存
    df_train_set_max.to_csv('./source/re_train_norm.csv', index=False)
    df_test.to_csv('./source/re_test_norm.csv', index=False)
    # df_val.to_csv('./source/re_val_norm.csv', index=False)


def fixDataset():
    dataset_df = pd.read_csv('./source/dataset.csv')
    error_df = pd.read_csv('./source/err.csv', index_col='mul_name')
    columns = error_df.columns
    for index, row in error_df.iterrows():
        for item in columns:
            dataset_df.loc[dataset_df.loc[:]['mul_name'] == index, item] = error_df.loc[index][item]
    dataset_df.to_csv('./source/dataset.csv', index=False)


if __name__ == '__main__':
    # getRetrain()
    # addRetrain2data()
    # normRetrain()
    # addRetrain2data()
    # normDataset()
    # normRetrain()
    fixDataset()
