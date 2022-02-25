import os
import shutil

import tensorflow as tf
import datetime
from tensorflow.keras import regularizers, Sequential, optimizers, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
import loaddata
from approx.fake_approx_convolutional import FakeApproxConv2D
import sys

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def vgg16cifra10app(app_src):
    img_input = tf.keras.Input(shape=(32, 32, 3), name='img_input')
    x = FakeApproxConv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1', mul_map_file=app_src)(
        img_input)

    x = FakeApproxConv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2', mul_map_file=app_src)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = FakeApproxConv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1', mul_map_file=app_src)(x)
    x = FakeApproxConv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2', mul_map_file=app_src)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = FakeApproxConv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1', mul_map_file=app_src)(x)
    x = FakeApproxConv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2', mul_map_file=app_src)(x)
    x = FakeApproxConv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3', mul_map_file=app_src)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = FakeApproxConv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1', mul_map_file=app_src,)(x)
    x = FakeApproxConv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2', mul_map_file=app_src)(x)
    x = FakeApproxConv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3', mul_map_file=app_src)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = FakeApproxConv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1', mul_map_file=app_src)(x)
    x = FakeApproxConv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2', mul_map_file=app_src)(x)
    x = FakeApproxConv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3', mul_map_file=app_src)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dense(10, activation='softmax',
                     name='predictions')(x)
    model = tf.keras.Model(img_input, x, name='vgg16')
    return model

mul_path = "./ApproxMul/"

def trainApp(mul_dirs, name, log_dir):
    app_src =mul_path + mul_dirs + "/Bin/" + name

    model = vgg16cifra10app(app_src)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    x_train, y_train, x_test, y_test, image_gen = loaddata.cifar10()
    batch_size = 128
    epochs = 100
    # image_gen.fit(x_train)

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, restore_best_weights=True)
    # model.summary()
    checkpoint_path = log_dir + "/" + name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + "/weights", save_weights_only=True,
                                                     save_best_only=True)
    print(name)
    if len(sys.argv) > 1 and sys.argv[1] == "image_gen":
        print("image_gen train")
        gen = image_gen.flow(x_train, y_train, batch_size=batch_size)
        model.fit_generator(generator=gen,
                            steps_per_epoch=50000 // batch_size,
                            epochs=epochs,
                            shuffle=True,
                            validation_data=(x_test, y_test),
                            callbacks=[cp_callback])
    else:
        print("original train")
        model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=[cp_callback])

    with open(checkpoint_path + "/acc.log", 'w') as acc_log:
        model.load_weights(checkpoint_path+"/weights")
        score = model.evaluate(x_test, y_test, verbose=1)

        acc_log.write("test_loss " + str(score[0]) + " " + "test_accuracy " + str(score[1]))
    print(log_dir)

if __name__ == '__main__':
    mul_dirs = os.listdir(mul_path)
    need_train = set()
    with open("mue_ED_0_50.log", "r") as f:
        for line in f.readlines():
            need_train.add(line.split()[0])

    for i in range(len(mul_dirs)):
        log_dir = "logstest/vgg16_app/" + mul_dirs[i]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        mul_lists = os.listdir(mul_path + mul_dirs[i] + "/Bin")
        trained_list = os.listdir(log_dir)
        for name in need_train:
            if name not in trained_list and name in mul_lists:
                trainApp(mul_dirs[i], name, log_dir)
            elif name in trained_list:
                if not os.path.isfile(log_dir + "/" + name + "/acc.log"):
                    shutil.rmtree(log_dir + "/" + name)
                    trainApp(mul_dirs[i], name, log_dir)





