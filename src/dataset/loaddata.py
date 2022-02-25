import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def cifar10():
    # CIFAR 10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    classes_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # plt.figure(figsize=(5, 3))
    # plt.subplots_adjust(hspace=0.1)
    # for n in range(15):
    #     plt.subplot(3, 5, n + 1)
    #     plt.imshow(x_train[n])
    #     plt.axis('off')
    # _ = plt.suptitle("geektutu.com CIFAR-10 Example")

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    x_train /= 255
    x_test /= 255
    image_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                         samplewise_center=False,  # set each sample mean to 0
                                         featurewise_std_normalization=False,  # divide inputs by std of dataset
                                         samplewise_std_normalization=False,  # divide each input by its std
                                         zca_whitening=False,  # apply ZCA whitening
                                         rotation_range=0,  # randomly rotate images in the range(degrees, 0 to 180)
                                         width_shift_range=0.1,
                                         # randomly shift images horizontally (fraction of total width)
                                         height_shift_range=0.1,
                                         # randomly shift images vertically (fraction of total height)
                                         horizontal_flip=True,  # randomly flip images
                                         vertical_flip=False,  # randomly flip images
                                         )

    # validation_datagen = image.ImageDataGenerator(rescale=1 / 255)
    # train_generator = image_gen_train.flow(x_train, y_train, batch_size=128)
    # test_generator = image_gen_test.flow(x_test, y_test, batch_size=128)

    image_gen.fit(x_train)
    return x_train, y_train, x_test, y_test, image_gen

def cifar100():
    # CIFAR 100 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()


    # plt.figure(figsize=(5, 3))
    # plt.subplots_adjust(hspace=0.1)
    # for n in range(15):
    #     plt.subplot(3, 5, n + 1)
    #     plt.imshow(x_train[n])
    #     plt.axis('off')
    # _ = plt.suptitle("geektutu.com CIFAR-10 Example")

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    x_train /= 255
    x_test /= 255
    image_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                         samplewise_center=False,  # set each sample mean to 0
                                         featurewise_std_normalization=False,  # divide inputs by std of dataset
                                         samplewise_std_normalization=False,  # divide each input by its std
                                         zca_whitening=False,  # apply ZCA whitening
                                         rotation_range=0,  # randomly rotate images in the range(degrees, 0 to 180)
                                         width_shift_range=0.1,
                                         # randomly shift images horizontally (fraction of total width)
                                         height_shift_range=0.1,
                                         # randomly shift images vertically (fraction of total height)
                                         horizontal_flip=True,  # randomly flip images
                                         vertical_flip=False,  # randomly flip images
                                         )

    # validation_datagen = image.ImageDataGenerator(rescale=1 / 255)
    # train_generator = image_gen_train.flow(x_train, y_train, batch_size=128)
    # test_generator = image_gen_test.flow(x_test, y_test, batch_size=128)

    image_gen.fit(x_train)
    return x_train, y_train, x_test, y_test, image_gen

def mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    # y_train = tf.squeeze(y_train, axis=1)
    # y_test = tf.squeeze(y_test, axis=1)
    x_train = tf.image.resize_with_crop_or_pad(x_train, 32, 32)
    x_test = tf.image.resize_with_crop_or_pad(x_test, 32, 32)
    x_train /= 255
    x_test /= 255
    image_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                         samplewise_center=False,  # set each sample mean to 0
                                         featurewise_std_normalization=False,  # divide inputs by std of dataset
                                         samplewise_std_normalization=False,  # divide each input by its std
                                         zca_whitening=False,  # apply ZCA whitening
                                         rotation_range=0,  # randomly rotate images in the range(degrees, 0 to 180)
                                         width_shift_range=0.1,
                                         # randomly shift images horizontally (fraction of total width)
                                         height_shift_range=0.1,
                                         # randomly shift images vertically (fraction of total height)
                                         horizontal_flip=True,  # randomly flip images
                                         vertical_flip=False,  # randomly flip images
                                         )

    image_gen.fit(x_train)
    return x_train, y_train, x_test, y_test, image_gen
