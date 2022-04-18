import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from error.data_process import processData

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import sys
sys.path.append('../')
from models import predict_model
from error.data_process import processDataSpec, processData
from evaluate import classify, regression
from models.predict_model import predictClassify, predictSpecClassify, predictRegression, predictSpectRegression
import matplotlib.pyplot as plt


def classifyMLP(df, feature_sel):
    y = df.loc[:,'classify']
    x = df.loc[:, feature_sel]

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
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
    model.fit(train_dataset, epochs=400, verbose=2)
    return model


def regressionMLP(df, feature_sel):
    y = df.loc[:, 'untrained_acc']
    x = df.loc[:, feature_sel]

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = dataset.shuffle(len(df)).batch(20)
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
    model.fit(train_dataset, epochs=200, verbose=2)

    return model


def plotDT(df, savename):
    plt.style.use(['science', 'ieee'])
    # df = df.sort_values(by='mape', ascending=True)
    df.to_csv('../result/csv/' + savename + '.csv')
    for index, data in df.iteritems():
        plt.plot(df.index, data.values, label=index)
    # plt.legend(label)
    plt.legend(loc='best')
    plt.xticks(rotation=300)

    plt.savefig('../result/' + savename + '.pdf', bbox_inches='tight')
    plt.show()


def mlpClaErrorModel():
    df = pd.read_csv('../../error/source/train_norm.csv')
    df = processDataSpec(df)
    # feature_index = ['WCRE', 'WCE', 'mue_ED0']
    feature_index = ['mue_ED0', 'mue_ED', 'ER']
    # feature_index = ['var_ED', 'var_RED', 'mue_RED']

    indexes = ['domain', 'vgg16mnist', 'resnet18mnist', 'resnet34mnist', 'vgg16cifar', 'resnet18cifar',
               'resnet34cifar', 'resnet34cifar100']
    dt_df = pd.DataFrame(index=indexes, columns=['top-1', 'top-2', 'recall-1', 'weight-tpr', 'macro-tpr'])

    for index in indexes:
        if index == 'domain':
            fixed_feature = ['net', 'dataset', 'concat']
            df_train = processData(df)
            model = classifyMLP(df_train, feature_index + fixed_feature)
            y, y_pre = predictClassify(model, feature_index + fixed_feature, 'mlp')

        else:

            df_train = df[df['concat'] == index]

            model = classifyMLP(df_train, feature_index)
            y, y_pre = predictSpecClassify(model, feature_index, 'mlp', index)
        res = classify.evaluation(y, y_pre)
        dt_df.loc[index, :] = res
    plotDT(dt_df, 'cla_mlp_model')

if __name__ == '__main__':
    mlpClaErrorModel()
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

# (0.07663883539536132, 0.9243362973604273) 200 256
