import os
import tensorflow as tf
from tensorflow.keras import Input, Model, optimizers
from resnet18approx import appBuildResNetI
import loaddata

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
# )
tf.config.experimental.set_memory_growth(gpus[0], True)


def appTest(name, file, result_path, mul_name, src=""):
    """
    when you change model, you should change 1.inputs, 2.loaddata function and 3.outputs function
    :param name: net name
    :param file: log to write
    :param result_path: dir path for file
    :param mul_name: appmul name
    :param src: appmul path
    :return:
    """
    inputs = Input(shape=(32, 32, 3))
    model = Model(inputs=inputs, outputs=appBuildResNetI(inputs, name, 100, src))
    # model = appBuildResNetI(src, inputs))
    model.load_weights(result_path + "weights")

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

    print('================================================================================')
    print('Testing trained model...')
    print(name, " ", src)
    x_train, y_train, x_test, y_test, image_gen = loaddata.cifar100()
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test top 5 acc:', score[2])
    file.write(mul_name + '  Test loss: ' + str(score[0]) + ' Test_top_5_acc ' + str(score[2]) + '  Test accuracy: ' + str(score[1]) + '\n')
    file.flush()


if __name__ == '__main__':
    # sequence test
    name = "resnet"
    net_type = "34"
    dataset = "cifar100"
    result_path = "./result/resnet34cifar100/"
    mul_path = "./ApproxMul/"
    model_type = name + net_type
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    mul_dirs = os.listdir(mul_path)

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
                appTest(model_type, f, result_path, mul_list[j], src)


    # name = "resnet"
    # net_type = "34"
    # dataset = "cifar"
    # result_path = "./result/" + name + dataset + net_type + "/"
    # mul_path = "/home/mo/workspace/pycharm/ApproxMul/"
    # model_type = name + net_type
    # if not os.path.exists(result_path):
    #     os.mkdir(result_path)
    #
    # mul_dirs = os.listdir(mul_path)
    #
    # tested = set()
    # for i in range(len(mul_dirs)):
    #     if os.path.exists(result_path + mul_dirs[i] + ".log"):
    #         with open(result_path + mul_dirs[i] + ".log", 'r') as file:
    #             for line in file.readlines():
    #                 content = line.split()
    #                 tested.add(content[0])
    # # print(tested)
    # for i in range(len(mul_dirs)):
    #     mul_list = os.listdir(mul_path + mul_dirs[i] + "/Bin/")
    #     mul_list.sort()
    #     with open(result_path + mul_dirs[i] + ".log", mode="a+") as f:
    #         for j in range(len(mul_list)):
    #             src = mul_path + mul_dirs[i] + '/Bin/' + mul_list[j]
    #             if mul_list[j] in tested:
    #                 continue
    #             print("last:", j / len(mul_list))
    #             print(src)
    #             appTest(model_type, f, result_path, mul_list[j], src)



