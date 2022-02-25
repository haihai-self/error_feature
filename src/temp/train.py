import os
from datetime import datetime

from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from loaddata import cifar10, mnist
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import optimizers

from resnet18 import buildResNetI
from resnet50 import buildResNetII
# from vgg import vgg16

batch_size = 128
epochs = 400

learning_rate = 0.1
regularizer = 1e-3
total_train_samples = 60000
total_test_samples = 10000
lr_decay_epochs = 1



def train(name):
    model = buildResNetI(name, 10)

    def scheduler(epoch):
        if epoch < epochs * 0.1:
            return learning_rate
        if epoch < epochs * 0.2:
            return learning_rate * 0.1
        if epoch < epochs * 0.4:
            return learning_rate * 0.01
        if epoch < epochs * 0.6:
            return learning_rate * 0.001
        return learning_rate * 0.0001
    logdir = "./logtest/" + name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    sgd = optimizers.SGD()
    adam = optimizers.Adam()
    reduce_lr = ReduceLROnPlateau()
    change_lr = LearningRateScheduler(scheduler)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, restore_best_weights=True)
    # board = TensorBoard(log_dir=logdir, histogram_freq=5, batch_size=batch_size, write_graph=True,
    #                             write_grads=True, write_images=True, embeddings_freq=5,
    #                             embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
    #                             update_freq='epoch')
    board = TensorBoard(log_dir=logdir, histogram_freq=1)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics = ['accuracy', keras.metrics.SparseCategoricalCrossentropy()]

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics='accuracy')

    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    # TensorBoard(log_dir="tflogs/{}".format(datetime.datetime.now().replace(microsecond=0).isoformat()))

    x_train, y_train, x_test, y_test, image_gen_train, image_gen_test = cifar10()

    model.fit(image_gen_train.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=image_gen_test.flow(x_test, y_test, batch_size=batch_size),
                        validation_steps=x_test.shape[0] // batch_size,
                        callbacks=[reduce_lr, early_stop, board])

    with open(logdir + "/acc.log", 'w') as acc_log:

        score = model.evaluate(image_gen_test.flow(x_test, y_test), verbose=1)

        acc_log.write("test loss" + str(score[0]) + " " + "test accuracy" + str(score[1]))
    print(logdir)
    model.save_weights(logdir + "/weights")


if __name__ == '__main__':
    train('resnet18')
