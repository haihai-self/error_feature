import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

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

    dataset = tf.data.Dataset.from_tensor_slices((x.values(), y.values()))
    train_dataset = dataset.shuffle(len(df)).batch(20)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(3, activation='softmax')
    ])

    # checkpoint_path = cpk_dir + 'weights'
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy',
    #                                                  save_weights_only=True,
    #                                                  save_best_only=True)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy()])
    model.fit(train_dataset, epochs=300, verbose=2)
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
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),

        # keras.layers.Dense(512, activation='relu'),
        # keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1, activation=keras.activations.sigmoid)
    ])
    model.compile(
        loss=keras.losses.mean_absolute_percentage_error,
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=[keras.metrics.mean_absolute_percentage_error])
    model.fit(train_dataset, epochs=400, verbose=2)

    return model

#
# def getProb():
#     feature_sel = ['mue_ED0', 'mue_ED', 'ER']
#     fixed_feature = ['net', 'dataset', 'concat']
#     feature_sel += fixed_feature
#     df1 = pd.read_csv('../../error/source/train_norm.csv', index_col='mul_name')
#     df1 = processData(df1)
#     df2 = pd.read_csv('../../error/source/test_norm.csv', index_col='mul_name')
#
#     model = regressionMLP(df1, feature_sel)
#     y, y_pre = predict_model.predictRegression(model, feature_sel, True)
#     df2.insert(0, column='y_pre', value=y_pre)
#     # df2.loc[:, 'y_pre'] = y_pre[:, 0]
#     df2.sort_values(by=['classify', 'y_pre', 'untrained_acc'], inplace=True, ascending=[True, False, False])
#     df2.to_csv('../result/csv/reg_pre.csv')

def buildErrorModel():
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')

    # 构建分类误差模型
    feature_index = ['mue_ED0', 'mue_ED', 'ER']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist',  'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])
    predict_model.claErrorModel(df_train, df_test, feature_index, indexes, classifyMLP, 'mlp', dt_df, 'cla_mlp_model')

    # # 构建回归误差模型
    feature_index =['mue_ED0', 'mue_ED', 'ER']
    dt_df = pd.DataFrame(index=indexes, columns=['MAPE', r'$\chi^2$'])
    predict_model.regErrorModel(df_train, df_test, feature_index, indexes, regressionMLP, 'mlp', dt_df, 'reg_mlp_model')

    # 构建retrain分类误差模型
    df_train = pd.read_csv('../../error/source/train_norm.csv')
    df_test = pd.read_csv('../../error/source/test_norm.csv')


    feature_index = ['WCRE', 'WCE', 'mue_ED0']
    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist',  'resnet18cifar', 'vgg16cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])


if __name__ == '__main__':
    buildErrorModel()
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

