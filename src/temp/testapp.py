import os
import tensorflow as tf
from tensorflow.keras import optimizers
import loaddata
from tensorflow.keras import Sequential, layers
import tensorflow as tf
from approx.fake_approx_convolutional import FakeApproxConv2D
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
# )
tf.config.experimental.set_memory_growth(gpus[0], True)

print(type(os.environ['LD_LIBRARY_PATH']))







def vggtest0(src):
    return tf.keras.Sequential([
        FakeApproxConv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1), mul_map_file=src),
        layers.AveragePooling2D(),
        FakeApproxConv2D(filters=16, kernel_size=(3, 3), activation='relu', mul_map_file=src),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])


def vggtest1(src=""):
    return Sequential([
        FakeApproxConv2D(8, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1), mul_map_file=src),
        FakeApproxConv2D(8, (3, 3), padding='same', activation="relu", mul_map_file=src),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        FakeApproxConv2D(16, (3, 3), padding='same', activation='relu', mul_map_file=src),
        FakeApproxConv2D(16, (3, 3), padding='same', activation="relu", mul_map_file=src),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        FakeApproxConv2D(32, (3, 3), padding='same', activation='relu', mul_map_file=src),
        FakeApproxConv2D(32, (3, 3), padding='same', activation="relu", mul_map_file=src),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])


def vggtest2(src=""):
    return Sequential([
        FakeApproxConv2D(8, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1),
                         kernel_regularizer=regularizers.l2(0.001), mul_map_file=src),
        FakeApproxConv2D(8, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        layers.MaxPooling2D(),

        FakeApproxConv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        FakeApproxConv2D(16, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        layers.MaxPooling2D(),

        FakeApproxConv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        FakeApproxConv2D(32, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])


def vggtest3(src=""):
    return Sequential([
        FakeApproxConv2D(8, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1),
                         kernel_regularizer=regularizers.l2(0.001), mul_map_file=src),
        FakeApproxConv2D(8, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        FakeApproxConv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        FakeApproxConv2D(16, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        FakeApproxConv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        FakeApproxConv2D(32, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(0.001),
                         mul_map_file=src),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])


weight_decay = 5e-4


def vgg16(src):
    model = Sequential()
    model.add(FakeApproxConv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3),
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(MaxPooling2D((2, 2)))

    model.add(FakeApproxConv2D(128, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(128, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(MaxPooling2D((2, 2)))

    model.add(FakeApproxConv2D(256, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(256, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(256, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(MaxPooling2D((2, 2)))

    model.add(FakeApproxConv2D(512, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(512, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(512, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(MaxPooling2D((2, 2)))

    model.add(FakeApproxConv2D(512, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(512, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))
    model.add(FakeApproxConv2D(512, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay), mul_map_file=src))

    model.add(Flatten())  # 2*2*512
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model




def appTest(file, result_path, mul_name, src=""):
    model = vgg16(src)
    model.load_weights(result_path + "weights")

    model.compile(optimizer='sgd',  #if vgg16 using sgd to optimizer else using adam
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print('================================================================================')
    print('Testing trained model...')
    print(src)
    x_train, y_train, x_test, y_test, image_gen_train, image_gen_test = loaddata.cifar10()
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    file.write(mul_name + '  Test loss: ' + str(score[0]) + '  Test accuracy: ' + str(score[1]) + '\n')
    file.flush()


if __name__ == '__main__':
    # sequence test
    name = "vgg"
    net_type = "16"
    dataset = "cifar"
    result_path = "./result/" + name + dataset + net_type + "/"
    mul_path = "/home/haihai/workspace/pycharm/ApproxMul/"
    model_type = name + net_type
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    mul_dirs = os.listdir(mul_path)

    #     f.writelines("hello" + src)
    tested = set()
    for i in range(len(mul_dirs)):
        if os.path.exists(result_path + mul_dirs[i] + ".log"):
            with open(result_path + mul_dirs[i] + ".log", 'r') as file:
                for line in file.readlines():
                    content = line.split()
                    tested.add(content[0])
    # print(tested)
    for i in range(len(mul_dirs)):
        mul_list = os.listdir(mul_path + mul_dirs[i] + "/Bin/")
        mul_list.sort()
        with open(result_path + mul_dirs[i] + ".log", mode="a+") as f:
            for j in range(len(mul_list)):
                src = mul_path + mul_dirs[i] + '/Bin/' + mul_list[j]
                if mul_list[j] in tested:
                    continue
                print("last:", j / len(mul_list))
                print(src)
                appTest(f, result_path, mul_list[j], src)
    print("finish test")
