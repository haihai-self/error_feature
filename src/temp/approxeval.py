import os
import tensorflow as tf
from tensorflow.keras import optimizers
import loaddata
from resnet18approx import appBuildResNetI


print(type(os.environ['LD_LIBRARY_PATH']))


def appTest(file, name, result_path, nb_classes, mul_name ,src=""):
    model = appBuildResNetI(name, nb_classes, src)
    model.load_weights(result_path + "weights")

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
    x_train, y_train, x_test, y_test, image_gen = loaddata.cifar100()

    model.build(input_shape=(None, 32, 32, 3))

    print('================================================================================')
    print('Testing trained model...')
    print(src)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    file.write(mul_name + '  Test loss: ' + str(score[0]) + '  Test accuracy: ' + str(score[1]) + '\n')

if __name__ == '__main__':
    # sequence test
    name = "resnet34"
    result_path = "./result/resnet34cifar100/"
    mul_path = "./ApproxMul/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    mul_dirs = os.listdir(mul_path)

        #     f.writelines("hello" + src)
    for i in range(len(mul_dirs)):
        mul_list = os.listdir(mul_path + mul_dirs[i] + "/Bin/")
        with open(result_path + mul_dirs[i] + ".log", mode="w") as f:
            for j in range(len(mul_list)):
                src = mul_path + mul_dirs[i] + '/Bin/' + mul_list[j]
                appTest(f, name, result_path, 100, mul_list[j] ,src)
    print("finish test")
