import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers, datasets, models, optimizers
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
from tensorflow.keras.datasets import cifar10
import os
from tensorflow.keras.optimizers import Adam

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
# )
tf.config.experimental.set_memory_growth(gpus[0], True)

import datetime
import loaddata

subtract_pixel_mean = True

name = 'resnet50'
batch_size = 32
epochs = 3

# 载入 CIFAR10 数据。
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 输入图像维度。
input_shape = x_train.shape[1:]

# 数据标准化。
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 如果使用减去像素均值
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
num_classes = 10


# 将类向量转换为二进制类矩阵。
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def basicBlock(inputs, num_fiters, strides=1):
    x = Conv2D(num_fiters[0], (1, 1), strides=strides, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(num_fiters[1], (3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(num_fiters[2], (1, 1), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    residual = Conv2D(num_fiters[2], (1, 1), strides=strides, padding='same', activation='relu')(inputs)
    residual = BatchNormalization()(residual)

    output = layers.add([x, residual])

    return output


def buildBlock(x, filter_num, block_num, strides):
    x = basicBlock(x, filter_num, strides=strides)
    for _ in range(1, block_num):
        x = basicBlock(x, filter_num, strides=1)
    return x


def ResNet(inputs, layer_dim, class_num):
    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='stem_pool')(x)

    filter_block0 = [64, 64, 256]
    filter_block1 = [128, 128, 512]
    filter_block2 = [256, 256, 1024]
    filter_block3 = [512, 512, 2048]

    x = buildBlock(x, filter_block0, layer_dim[0], strides=1)
    x = buildBlock(x, filter_block1, layer_dim[1], strides=2)
    x = buildBlock(x, filter_block2, layer_dim[2], strides=2)
    x = buildBlock(x, filter_block3, layer_dim[3], strides=2)

    x = GlobalAveragePooling2D(name='top_layer_pool')(x)
    x = Dense(class_num, activation="softmax",  kernel_initializer='he_normal')(x)

    return x


def buildResNetII(inputs, name, nb_clssses):
    nets = {'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3]}
    return ResNet(inputs, nets[name], nb_clssses)



def lr_schedule(epoch):
    """学习率调度

    学习率将在 80, 120, 160, 180 轮后依次下降。
    他作为训练期间回调的一部分，在每个时期自动调用。

    # 参数
        epoch (int): 轮次

    # 返回
        lr (float32): 学习率
    """
    lr = 1e-3
    if epoch > 0.9 * epochs:
        lr *= 0.5e-3
    elif epoch > 0.8 * epochs:
        lr *= 1e-3
    elif epoch > 0.6 * epochs:
        lr *= 1e-2
    elif epoch > 0.4 * epochs:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

model_type = 5
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


if __name__ == '__main__':


    inputs = Input(shape=(32, 32, 3))
    model = Model(inputs=inputs, outputs=buildResNetII(inputs, name, 10))

    # model = buildResNetII(name, 10)
    # model.build(input_shape=(None, 32, 32, 3))

    model.summary()
    # tf.keras.utils.plot_model(model, "resnet50.png", show_shapes=True)

    log_dir = "logstest/" + name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        # 在整个数据集上将输入均值置为 0
        featurewise_center=False,
        # 将每个样本均值置为 0
        samplewise_center=False,
        # 将输入除以整个数据集的 std
        featurewise_std_normalization=False,
        # 将每个输入除以其自身 std
        samplewise_std_normalization=False,
        # 应用 ZCA 白化
        zca_whitening=False,
        # ZCA 白化的 epsilon 值
        zca_epsilon=1e-06,
        # 随机图像旋转角度范围 (deg 0 to 180)
        rotation_range=0,
        # 随机水平平移图像
        width_shift_range=0.1,
        # 随机垂直平移图像
        height_shift_range=0.1,
        # 设置随机裁剪范围
        shear_range=0.,
        # 设置随机缩放范围
        zoom_range=0.,
        # 设置随机通道切换范围
        channel_shift_range=0.,
        # 设置输入边界之外的点的数据填充模式
        fill_mode='nearest',
        # 在 fill_mode = "constant" 时使用的值
        cval=0.,
        # 随机翻转图像
        horizontal_flip=True,
        # 随机翻转图像
        vertical_flip=False,
        # 设置重缩放因子 (应用在其他任何变换之前)
        rescale=None,
        # 设置应用在每一个输入的预处理函数
        preprocessing_function=None,
        # 图像数据格式 "channels_first" 或 "channels_last" 之一
        data_format=None,
        # 保留用于验证的图像的比例 (严格控制在 0 和 1 之间)
        validation_split=0.0)

    # 计算大量的特征标准化操作
    # (如果应用 ZCA 白化，则计算 std, mean, 和 principal components)。
    datagen.fit(x_train)

    # 在由 datagen.flow() 生成的批次上拟合模型。
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=2, workers=4,
                        callbacks=callbacks)

    model.save_weights(log_dir + "/weights")
    with open(log_dir + "/acc.log", 'w') as acc_log:
        score = model.evaluate(x_test, y_test, verbose=1)

        acc_log.write("test_loss " + str(score[0]) + " " + "test_accuracy " + str(score[1]))
    print(log_dir)
