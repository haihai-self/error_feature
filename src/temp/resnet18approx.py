import tensorflow as tf
from approx.fake_approx_convolutional import FakeApproxConv2D
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, AveragePooling2D, Input
from approx.fake_approx_convolutional import FakeApproxConv2D



def basicBlock(block_inputs, num_fiters, strides=1, src=""):
    # if src == "":
    #     print("error")
    #     exit(0)
    x = FakeApproxConv2D(num_fiters, (3, 3), strides=strides, padding='same', activation='relu', mul_map_file=src)(block_inputs)
    x = BatchNormalization()(x)

    x = FakeApproxConv2D(num_fiters, (3, 3), strides=1, padding='same', activation='relu', mul_map_file=src)(x)
    x = BatchNormalization()(x)
    if strides != 1:
        block_inputs = Conv2D(num_fiters, (1, 1), strides=strides)(block_inputs)

    block_output = layers.add([x, block_inputs])
    return block_output


def buildBlock(x, filter_num, block_num, strides, src=""):
    x = basicBlock(x, filter_num, strides, src=src)
    for _ in range(1, block_num):
        x = basicBlock(x, filter_num, strides=1, src=src)

    return x


def AppResNet(inputs, layer_dim, class_num, src=""):
    x = FakeApproxConv2D(64, (3, 3), strides=1, padding='same', activation='relu', mul_map_file=src)(inputs)
    x = BatchNormalization()(x)
    # x = MaxPool2D((2, 2), strides=1, padding='same')(x)

    x = buildBlock(x, 64, layer_dim[0], strides=1, src=src)
    x = buildBlock(x, 128, layer_dim[1], strides=2, src=src)
    x = buildBlock(x, 256, layer_dim[2], strides=2, src=src)
    x = buildBlock(x, 512, layer_dim[3], strides=2, src=src)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    x = Dense(class_num)(x)

    return x


def appBuildResNetI(inputs, name, nb_clssses=100, src=""):
    nets = {'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3]
            }
    return AppResNet(inputs, nets[name], nb_clssses, src)
