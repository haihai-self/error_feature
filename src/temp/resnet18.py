import tensorflow as tf
from tensorflow.keras import layers, datasets, models, optimizers, Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, AveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
import datetime
import loaddata
import sys
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)


def basicBlock(block_inputs, num_fiters, strides=1):
    x = Conv2D(num_fiters, (3, 3), strides=strides, padding='same', activation='relu')(block_inputs)
    x = BatchNormalization()(x)

    x = Conv2D(num_fiters, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    if strides != 1:
        block_inputs = Conv2D(num_fiters, (1, 1), strides=strides)(block_inputs)

    block_output = layers.add([x, block_inputs])
    return block_output


def buildBlock(x, filter_num, block_num, strides):
    x = basicBlock(x, filter_num, strides)
    for _ in range(1, block_num):
        x = basicBlock(x, filter_num, strides=1)

    return x


def ResNet(inputs, layer_dim, class_num):
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    # x = MaxPool2D((2, 2), strides=1, padding='same')(x)

    x = buildBlock(x, 64, layer_dim[0], strides=1)
    x = buildBlock(x, 128, layer_dim[1], strides=2)
    x = buildBlock(x, 256, layer_dim[2], strides=2)
    x = buildBlock(x, 512, layer_dim[3], strides=2)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    x = Dense(class_num)(x)

    return x


class BasicBlock(layers.Layer):
    def __init__(self, num_filters, strides=1):
        super().__init__()

        self.conv1 = layers.Conv2D(num_filters, (3, 3), strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.conv2 = layers.Conv2D(num_filters, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.identity_layer = layers.Conv2D(num_filters, (1, 1), strides=strides)
        self.strides = strides


    def call(self, inputs, training=None):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.strides != 1:
            identity_out = self.identity_layer(inputs)
        else:
            identity_out = inputs
        output = layers.add([output, identity_out])
        output = tf.nn.relu(output)

        return output


class ResNet1(models.Model):
    def __init__(self, layer_dim, class_num):  # layer_dim res18:[2,2,2,2] or res34[3,4,6,3]
        super().__init__()
        self.pre_process = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            # layers.MaxPool2D((2, 2), strides=1, padding='same')
        ])

        self.resblock0 = self.build_resblock(64, layer_dim[0], strides=1)
        self.resblock1 = self.build_resblock(128, layer_dim[1], strides=2)
        self.resblock2 = self.build_resblock(256, layer_dim[2], strides=2)
        self.resblock3 = self.build_resblock(512, layer_dim[3], strides=2)

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(class_num)

    def build_resblock(self, filter_num, basic_block_num, strides):
        res_block = tf.keras.Sequential()
        res_block.add(BasicBlock(filter_num, strides))
        for _ in range(1, basic_block_num):
            basic_block = BasicBlock(filter_num, strides=1)
            res_block.add(basic_block)

        return res_block

    def call(self, inputs, training=None):
        output = self.pre_process(inputs)  # [b, 64, h, w]

        output = self.resblock0(output)  # [b, 64, h, w]
        output = self.resblock1(output)  # [b, 128, h, w]
        output = self.resblock2(output)  # [b, 256, h, w]
        output = self.resblock3(output)  # [b, 512, h, w]

        output = self.avg_pool(output)  # [b, 512, 1,1]
        output = self.dense(output)  # [b, class_num]

        return output


def buildResNetI(inputs, name, nb_clssses):
    nets = {'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3]
            }
    return ResNet(inputs, nets[name], nb_clssses)


def buildResNet(name, nb_clssses):
    nets = {'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3]
            }
    return ResNet1(nets[name], nb_clssses)


if __name__ == '__main__':
    name = 'resnet34'
    batch_size = 256
    epochs = 400

    inputs = Input(shape=(32, 32, 3))
    model = Model(inputs=inputs, outputs=buildResNetI(inputs, name, 100))

    # model = buildResNetII(name, 10)
    # model.build(input_shape=(None, 32, 32, 3))

    model.summary()
    # tf.keras.utils.plot_model(model, "resnet18.png", show_shapes=True)

    log_dir = "logstest/resnet34cifar100/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # reduce_lr = ReduceLROnPlateau()
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, restore_best_weights=True)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # optim = optimizers.Adam()
    # model.compile(optimizer=optim,
    #               loss=loss,
    #               metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
    x_train, y_train, x_test, y_test, image_gen = loaddata.cifar100()

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

