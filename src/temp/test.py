import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow.keras.applications

def conv2d_test(inputx):
    x = layers.Conv2D(filters=3, kernel_size=(2, 2), input_shape=(4, 4, 1), strides=1, kernel_initializer='ones')(
        inputx)
    return x


def test():
    data_test = np.random.randint(1, 3, (1, 4, 4, 1))
    print(data_test)
    x = conv2d_test(data_test)
    print(x)

IMG_SHAPE = (32, 32, 3)
base_model = app.vgg16(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
base_model.summary()