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
from evaluate import classify, regression



def classifyMLP(df, feature_sel):
    y = df.loc[:,'classify']
    x = df.loc[:, feature_sel]

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = dataset.shuffle(len(df)).batch(20)

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    # checkpoint_path = cpk_dir + 'weights'
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy',
    #                                                  save_weights_only=True,
    #                                                  save_best_only=True)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy()])
    model.fit(train_dataset, epochs=200, verbose=2)
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

if __name__ == '__main__':
    df = pd.read_csv('../../error/source/train_norm.csv')
    df = processData(df)
    sel_feature = ['mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                   'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'net', 'dataset', 'concat', 'single-sided', 'zero-error']
    # model = classifyMLP(df, sel_feature)
    # y, y_pre = predict_model.predictClassify(model, sel_feature, 'mlp')
    # print(classify.evaluation(y, y_pre))

    model = regressionMLP(df, sel_feature)
    y, y_pre = predict_model.predictRegression(model, sel_feature)
    print(regression.evaluation(y, y_pre))

# (0.07663883539536132, 0.9243362973604273) 200 256
