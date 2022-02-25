import tensorflow as tf
import datetime
import tensorflow.keras as keras
from tensorflow.keras import regularizers, Sequential, optimizers, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
import loaddata
import numpy as np
import tensorflow.keras.applications
import sys

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

weight_decay = 5e-4


def vgg16cifra10():
    img_input = tf.keras.Input(shape=(32, 32, 3), name='img_input')
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dense(10, activation='softmax',
                     name='predictions')(x)
    model = tf.keras.Model(img_input, x, name='vgg16')
    return model


if __name__ == '__main__':
    model = vgg16cifra10()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    x_train, y_train, x_test, y_test, image_gen = loaddata.cifar10()
    batch_size = 128
    epochs = 100
    # image_gen.fit(x_train)

    name = "vgg16"
    log_dir = "logstest/" + name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, restore_best_weights=True)
    model.summary()
    # plot_model(model, 'vgg16.png', show_shapes=True, show_layer_names=True)
    checkpoint_path = log_dir + "/weights"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_weights_only=True, save_best_only=True)
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

    with open(log_dir + "/acc.log", 'w') as acc_log:
        model.load_weights(checkpoint_path)
        score = model.evaluate(x_test, y_test, verbose=1)

        acc_log.write("test_loss " + str(score[0]) + " " + "test_accuracy " + str(score[1]))
    print(log_dir)

