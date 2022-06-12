import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import sys
sys.path.append('../')
from models import predict_model

def classifyMLP(df, feature_sel):
    """
    MLP回归模型
    :param df: train data DataFrame 数据结构
    :param feature_sel: 需要选择的特征
    :return: MLP分类模型
    """
    y = df.loc[:,'classify']
    x = df.loc[:, feature_sel]

    dataset = tf.data.Dataset.from_tensor_slices((x.values, y.values))
    train_dataset = dataset.shuffle(len(df)).batch(20)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', dtype=tf.float32),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu', dtype=tf.float32),

        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu', dtype=tf.float32),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(3, activation='softmax')
    ])

    # checkpoint_path = '../result/cpk/' + 'weights'
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy',
    #                                                  save_weights_only=True,
    #                                                  save_best_only=True)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=False)])
    model.fit(train_dataset, epochs=200, verbose=2)
    return model


def regressionMLP(df, feature_sel):
    """
    MLP回归模型
    :param df: train DataFrame 数据
    :param feature_sel: list 需要用到的特征
    :return: MLP模型
    """
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_sel]

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = dataset.shuffle(len(df)).batch(20)
    # 模型参数修改
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', dtype=tf.float32),
        keras.layers.Dense(256, activation='relu', dtype=tf.float32),
        keras.layers.Dense(512, activation='relu', dtype=tf.float32),
        keras.layers.Dense(1024, activation='relu', dtype=tf.float32),
        keras.layers.Dense(128, activation='relu', dtype=tf.float32),
        keras.layers.Dense(1, activation=keras.activations.sigmoid)
    ])
    model.compile(
        loss=keras.losses.mean_absolute_percentage_error,
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=[keras.metrics.mean_absolute_percentage_error])
    model.fit(train_dataset, epochs=200, verbose=2)

    return model


def mlpClaRetrain():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')
    feature_rank = ['mue_ED0', 'mue_ED', 'ER', 'RMS_ED', 'NMED', 'var_ED0', 'var_ED',
       'mue_RED', 'zero-error', 'RMS_RED', 'var_RED', 'var_ARED', 'WCE', 'mue_ARED',
       'single-sided', 'WCRE']
    for i in range(1, len(feature_rank) + 1):
        feature_index = feature_rank[0:i]
        indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
                   'resnet34cifar', 'resnet34cifar100']
        dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
        predict_model.claErrorModel(df_train, df_test, feature_index, indexes, classifyMLP, 'mlp', dt_df,
                                    'cla_mlp_model' + str(i))

def mlpClaZeroout():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')
    feature_rank = ['mue_ED0', 'mue_ED', 'ER', 'RMS_ED', 'NMED', 'var_ED0', 'var_ED',
                    'mue_RED', 'zero-error', 'RMS_RED', 'var_RED', 'var_ARED', 'WCE', 'mue_ARED',
                    'single-sided', 'WCRE']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    predict_model.claZeroModel(df_train, df_test, feature_rank, indexes, classifyMLP)




def mlpRegRetrain():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')
    feature_rank = ['mue_ED0', 'mue_ED', 'ER', 'RMS_ED', 'var_ED0', 'var_ED', 'zero-error',
       'WCE', 'RMS_RED', 'mue_ARED', 'var_ARED', 'var_RED', 'mue_RED', 'NMED',
       'single-sided', 'WCRE']
    for i in range(15, len(feature_rank) + 1):
        feature_index = feature_rank[0:i]
        indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
                   'resnet34cifar', 'resnet34cifar100']
        dt_df = pd.DataFrame(index=indexes, columns=['MAPE', r'$\chi^2$'])
        predict_model.regErrorModel(df_train, df_test, feature_index, indexes, regressionMLP, 'mlp', dt_df,
                                    'reg_mlp_model'+str(i))

def buildErrorModel():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')

    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    indexes = ['domain']

    # 构建分类误差模型
    # feature_index = ['mue_ED0', 'mue_ED', 'ER']
    # dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    # predict_model.claErrorModel(df_train, df_test, feature_index, indexes, classifyMLP, 'mlp', dt_df, 'cla_mlp_model')

    # # 构建回归误差模型
    feature_index =['mue_ED0', 'mue_ED', 'ER']
    dt_df = pd.DataFrame(index=indexes, columns=['MAPE', r'$\chi^2$'])
    predict_model.regErrorModel(df_train, df_test, feature_index, indexes, regressionMLP, 'mlp', dt_df, 'reg_mlp_model')

    # 构建retrain分类误差模型
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')


    feature_index = ['WCRE', 'WCE', 'mue_ED0']
    indexes = indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar','resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    indexes = ['domain']
    df_plot = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])

def getlegend():
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    line = ['-', '--', ':', '-.', '--', ':', '-.', '--']
    colors = ['k', 'b', 'b', 'b', 'g', 'g', 'g', 'r']
    f = lambda m, c: plt.plot([], [], linestyle=m, color=c)[0]
    handles = [f(line[i], colors[i]) for i in range(8)]
    labels = indexes
    legend = plt.legend(handles, labels, ncol=4, framealpha=1, frameon=False, bbox_to_anchor=(0, -2))

    def export_legend(legend, filename="legend.pdf"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    export_legend(legend)
    plt.show()


if __name__ == '__main__':
    getlegend()
    # mlpClaRetrain()
    # exClaRetrain()
    # mlpRegRetrain()
    # buildErrorModel()
    # mlpClaZeroout()
    # mlpClaErrorModel()
    # getProb()
    # df = pd.read_csv('../../error/source/train_norm.csv')
    # df = processData(df)
    # sel_feature = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
    #                'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'net', 'dataset', 'concat', 'single-sided', 'zero-error']
    # # model = classifyMLP(df, sel_feature)
    # # y, y_pre = predict_model.predictClassify(model, sel_feature, 'mlp')
    # # print(classify.evaluation(y, y_pre))
    #
    # model = regressionMLP(df, sel_feature)
    # y, y_pre = predict_model.predictRegression(model, sel_feature)
    # print(regression.evaluation(y, y_pre))

