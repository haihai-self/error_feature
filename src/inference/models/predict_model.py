import sys

sys.path.append('../../')
import numpy as np
import pandas as pd
from error.data_process import processData, processDataSpec
import matplotlib.pyplot as plt
from evaluate import classify, regression

def predictClassify(model, feature_index, model_name, df):
    """
    domain模型--预测分类模型对应的准确率
    :param model: 模型
    :param feature_index: 训练模型所用到的特征
    :param model_name: 模型对应的名字，MLP SVM DT RF
    :return: y y_pre(top1, top2) y_pre 为2维
    """
    # 读取测试文件

    x = df.loc[:, feature_index]
    y = df.loc[:, 'classify']

    # 计算前两类预测值
    # x = model._validate_X_predict(x, True)
    if model_name == 'svm':
        proba = model.decision_function(x)
    elif model_name == 'mlp':
        proba = model.predict(x)
    else:
        proba = model.predict_proba(x)
    top1 = np.argmax(proba, axis=1)
    for i in range(len(proba)):
        proba[i][top1[i]] = -1
    top2 = np.argmax(proba, axis=1)
    y_pre = np.stack([top1, top2], axis=1)
    y = np.array(y)
    return y, y_pre

def predictClassifyZero(model, feature_index, model_name, df, type='domain'):
    """
    domain模型--预测分类模型对应的准确率
    :param model: 模型
    :param feature_index: 训练模型所用到的特征
    :param model_name: 模型对应的名字，MLP SVM DT RF
    :return: y y_pre(top1, top2) y_pre 为2维
    """
    # 读取测试文件
    if type == 'domain':
        all_feature = ['mue_ED0', 'mue_ED', 'ER', 'RMS_ED', 'NMED', 'var_ED0', 'var_ED',
                    'mue_RED', 'zero-error', 'RMS_RED', 'var_RED', 'var_ARED', 'WCE', 'mue_ARED',
                    'single-sided', 'WCRE', 'net', 'dataset', 'concat']
    else:
        all_feature = ['mue_ED0', 'mue_ED', 'ER', 'RMS_ED', 'NMED', 'var_ED0', 'var_ED',
                       'mue_RED', 'zero-error', 'RMS_RED', 'var_RED', 'var_ARED', 'WCE', 'mue_ARED',
                       'single-sided', 'WCRE']
    x = df.loc[:, all_feature]
    for i in range(len(all_feature)):
        if all_feature[i] in feature_index:
            continue
        else:
            x.loc[:, all_feature[i]] = 0

    y = df.loc[:, 'classify']

    # 计算前两类预测值
    # x = model._validate_X_predict(x, True)
    if model_name == 'svm':
        proba = model.decision_function(x)
    elif model_name == 'mlp':
        proba = model.predict(x)
    else:
        proba = model.predict_proba(x)
    top1 = np.argmax(proba, axis=1)
    for i in range(len(proba)):
        proba[i][top1[i]] = -1
    top2 = np.argmax(proba, axis=1)
    y_pre = np.stack([top1, top2], axis=1)
    y = np.array(y)
    return y, y_pre

def predictRegression(model, feature_index, df):
    """
    domain模型--预测回归模型准确率
    :param model: 模型
    :param feature_index: 训练模型所用到的特征
    :param model_name: 模型对应的名字，MLP SVM DT RF
    :return: 预测数据
    """
    df = processData(df)
    x = df.loc[:, feature_index]
    y = df.loc[:, 'untrained_acc']
    y_pre = model.predict(x)

    return y, y_pre

def predictSpectRegression(model, feature_index, model_name, spec, test_or_val=True):
    """
    spec模型---预测回归模型准确率
    :param model: 模型
    :param feature_index: 训练模型所用到的特征
    :param model_name: 模型对应的名字，MLP SVM DT RF
    :param spec: 是哪一种应用，vgg16mnist, resnet18cifar, resnet34cifar100等
    :return: y y_pre 都是一维
    """
    # 读取测试文件
    if test_or_val:
        df = pd.read_csv('../../error/source/test_norm.csv')
    else:
        df = pd.read_csv('../../error/source/val_norm.csv')
    df = processDataSpec(df)

    x = df.loc[df['concat'] == spec, feature_index]
    y = df.loc[df['concat'] == spec, 'untrained_acc']

    y_pre = model.predict(x)
    y = np.array(y)

    return y, y_pre

def plotDF(df, savename):
    """
    绘制df数据折线图, df中每一行为一条线
    :param df: DataFrame 数据结构，
    :param savename:保存pdf的文件名
    """
    plt.style.use(['science', 'ieee'])
    df.to_csv('../result/csv/' + savename + '.csv')

    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    plt.legend(loc='best')
    plt.xticks(rotation=300)

    plt.savefig('../result/' + savename + '.pdf', bbox_inches='tight')
    plt.close()

def claZeroModel(df_train, df_test, feature_rank, indexes, model):
    """
        构建分类误差模型
        :param df_train: 训练数据
        :param df_test: 测试数据
        :param feature_index: 训练需要的特征
        :param indexes: 需要构建的应用名称
        :param model: 需要训练的模型
        :param model_name: 模型的名称, svm, mlp等
        :return:
        """
    fixed_feature = ['net', 'dataset', 'concat']
    model_dict = {}
    for i in range(len(indexes)):
        if indexes[i] == 'domain':
            df = processData(df_train.copy())
            trained_model = model(df, feature_rank + fixed_feature)
            model_dict[indexes[i]] = trained_model
        else:
            df = processDataSpec(df_train.loc[df_train.loc[:, 'concat'] == indexes[i], :].copy())
            trained_model = model(df, feature_rank)
            model_dict[indexes[i]] = trained_model

    df_plot = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    feature_rank_len = len(feature_rank)
    for i in range(feature_rank_len):
        feature_index = feature_rank[0:i+1]
        savename = 'cla_mlp_model'+str(i+1)
        for index in indexes:
            if index == 'domain':
                df = processData(df_test.copy())
                y, y_pre = predictClassifyZero(model_dict[index], feature_index + fixed_feature, 'mlp', df)
            else:
                df = processDataSpec(df_test.loc[df_test.loc[:, 'concat'] == index, :].copy())
                y, y_pre = predictClassifyZero(model_dict[index], feature_index, 'mlp', df, 'spec')
            res = classify.evaluation(y, y_pre)
            print(res)
            df_plot.loc[index, :] = res
        df_plot.to_csv('../result/csv/' + savename + '.csv')
        # plotDF(df_plot, pdf_name)

def claErrorModel(df_train, df_test, feature_index, indexes,  model, model_name, df_plot, pdf_name):
    """
    构建分类误差模型
    :param df_train: 训练数据
    :param df_test: 测试数据
    :param feature_index: 训练需要的特征
    :param indexes: 需要构建的应用名称
    :param model: 需要训练的模型
    :param model_name: 模型的名称, svm, mlp等
    :param df_plot: 绘制数据使用的DataFrame
    :param pdf_name: 保存pdf的名称
    :return:
    """

    for index in indexes:
        if index == 'domain':
            fixed_feature = ['net', 'dataset', 'concat']
            df = processData(df_train.copy())
            trained_model = model(df, feature_index + fixed_feature)
            df = processData(df_test.copy())
            y, y_pre = predictClassify(trained_model, feature_index + fixed_feature, model_name, df)
            # 保存预测值
            df = df_test.copy()
            df.insert(5, column='y_pre', value=y_pre[:, 0])
            df.sort_values(by=['y_pre', 'untrained_acc'], inplace=True, ascending=[True, False])
        else:
            df = processDataSpec(df_train.loc[df_train.loc[:, 'concat'] == index, :].copy())
            trained_model = model(df, feature_index)
            df = processDataSpec(df_test.loc[df_test.loc[:, 'concat'] == index, :].copy())
            y, y_pre = predictClassify(trained_model, feature_index, model_name, df)
        res = classify.evaluation(y, y_pre)
        df_plot.loc[index, :] = res
    plotDF(df_plot, pdf_name)

def regErrorModel(df_train, df_test, feature_index, indexes,  model, model_name, df_plot, pdf_name):
    """
    构建回归模型
    :param df_train: 训练数据
    :param df_test: 测试数据
    :param feature_index: 训练需要的特征
    :param indexes: 需要构建的应用名称
    :param model: 需要训练的模型
    :param model_name: 模型的名称, svm, mlp等
    :param df_plot: 绘制数据使用的DataFrame
    :param pdf_name: 保存pdf的名称
    :return:
    """
    for index in indexes:
        if index == 'domain':
            fixed_feature = ['net', 'dataset', 'concat']
            df = processData(df_train.copy())
            trained_model = model(df, feature_index + fixed_feature)
            df = processData(df_test.copy())
            y, y_pre = predictRegression(trained_model, feature_index + fixed_feature, df)
            df = df_test.copy()
            df.insert(0, column='y_pre', value=y_pre)
            df.sort_values(by=['classify', 'y_pre', 'untrained_acc'], inplace=True, ascending=[True, False, False])
        else:
            df = processDataSpec(df_train.loc[df_train.loc[:, 'concat'] == index, :].copy())
            trained_model = model(df, feature_index)
            df = processDataSpec(df_test.loc[df_test.loc[:, 'concat'] == index, :].copy())
            y, y_pre = predictRegression(trained_model, feature_index, df)
        res = regression.evaluation(y, y_pre)
        df_plot.loc[index, :] = res
    plotDF(df_plot, pdf_name)