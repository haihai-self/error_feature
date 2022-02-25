import functools

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name="untrained_acc",
        num_epochs=1,
        ignore_errors=True,
        na_value="-1"
    )
    return dataset


def process_continuous_data(mean, data):
    # 标准化数据
    data = tf.cast(data, tf.float32) * 1 / (2 * mean)
    return tf.reshape(data, [-1, 1])


def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2


if __name__ == '__main__':
    train_file_path = "./train_data.csv"
    test_file_path = "./test_data.csv"
    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    CATEGORIES = {
        'net': ['resnet18', 'resnet34', 'vgg16'],
        'dataset': ['mnist', 'cifar'],
        'concat': ['resnet18mnist', 'resnet18cifar', 'resnet34mnist', 'resnet34cifar', 'vgg16mnist', 'vgg16cifar']
    }

    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    MEANS = {
        'mue_ED0': 224.8647,
        'var_ED0': 38638.14,
        'mue_ED': 301.095,
        'NMED': 0.026522,
        'var_ED': 36933.637,
        'mue_RED': 2537.856,
        'var_RED': 5089.6794,
        'mue_ARED': 2537.8563,
        'var_ARED': 5089.6794,
        'RMS_ED': 562.6553,
        'RMS_RED': 2544.961,
        'ER': 0.816015,
        'WCE': 1461.975
    }
    numerical_columns = []

    for feature in MEANS.keys():
        num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data,
                                                                                            MEANS[feature]))
        numerical_columns.append(num_col)

    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)
    train_data = raw_train_data.shuffle(500)
    test_data = raw_test_data
    # model = tf.keras.Sequential([
    #     preprocessing_layer,
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.Dense(2048, activation='relu'),
    #     tf.keras.layers.Dense(4096, activation='relu'),
    #     tf.keras.layers.Dense(4096, activation='relu'),
    #
    #     # tf.keras.layers.Dense(1024, activation='relu'),
    #     # tf.keras.layers.Dense(2048, activation='relu'),
    #     # tf.keras.layers.Dense(2048, activation='relu'),
    #     # tf.keras.layers.Dense(4096, activation='relu'),
    #     # tf.keras.layers.Dense(4096, activation='relu'),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #
    #     tf.keras.layers.Dense(1, activation=keras.activations.sigmoid),
    # ])
    x = preprocessing_layer(train_data)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    output = tf.keras.layers.Dense(1, activation=keras.activations.sigmoid)(x)

    model = keras.Model(inpus=[v for v in train_data.values()], output=output)

    checkpoint_path = './regressing/16_23'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + "/weights", save_weights_only=True,
                                                     save_best_only=True)
    model.compile(
        loss=keras.losses.mean_absolute_percentage_error,
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=keras.metrics.mean_absolute_percentage_error)

    model.fit(train_data, epochs=1000, validation_data=test_data, )

    test_loss, test_accuracy = model.evaluate(test_data)

    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
