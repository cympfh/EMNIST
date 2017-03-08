import os

import click
import numpy
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Convolution2D, Dense, Flatten, Input, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import np_utils
from PIL import Image

from dataset import load_emnist

Y = 10
M = 32


def make_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('f') / 255.0  # (60000, 28, 28)
    x_test = x_test.astype('f') / 255.0
    y_train = np_utils.to_categorical(y_train, Y)  # int -> one-of-vector
    y_test = np_utils.to_categorical(y_test, Y)
    return x_train, y_train, x_test, y_test


def make_model():

    def Encoder():
        x = Input(shape=(28, 28))
        h = Reshape((28, 28, 1))(x)
        h = BatchNormalization()(h)
        h = Convolution2D(8, 5, 5, subsample=(2, 2), activation='relu')(h)
        h = BatchNormalization()(h)
        h = Convolution2D(16, 5, 5, subsample=(2, 2), activation='relu')(h)
        h = BatchNormalization()(h)
        h = Flatten()(h)
        z = Dense(M)(h)
        model = Model(input=x, output=z)
        return model

    def Decoder():
        model = Sequential(name='decoder')
        model.add(Dense(28 * 28 * 8, input_shape=(M,)))
        model.add(Reshape((28, 28, 8)))
        model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(4, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(1, 3, 3, border_mode='same', activation='relu'))
        model.add(Reshape((28, 28)))
        return model

    x = Input(shape=(28, 28))
    z = Encoder()(x)
    y = Decoder()(z)
    model = Model(input=x, output=y)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


@click.group()
def cli():
    pass


@cli.command()
def train():

    x_train, _, x_test, _ = load_emnist(dim=2, norm=True)
    model = make_model()

    os.path.exists('.weights') or os.mkdir('.weights')
    save_path = '.weights/autoencoder.hdf5'
    cp = ModelCheckpoint(filepath=save_path, verbose=2, save_best_only=True, save_weights_only=True)

    model.fit(x_train, x_train, validation_data=(x_test, x_test), batch_size=20, nb_epoch=100, callbacks=[cp])


@cli.command()
def test():

    _, _, x_test, y_test = load_emnist(dim=2, norm=True)
    model = make_model()
    save_path = '.weights/autoencoder.hdf5'
    model.load_weights(save_path)

    x_tensor = model.layers[1].layers[0].input
    z_tensor = model.layers[1].layers[-1].output
    encoder = K.function([x_tensor, K.learning_phase()], [z_tensor])

    z_tensor = model.layers[2].layers[0].input
    y_tensor = model.layers[2].layers[-1].output
    decoder = K.function([z_tensor], [y_tensor])

    # morphing
    z = encoder([x_test[:100], 0])[0]

    for k in range(10):
        _z = numpy.zeros((10, 32)).astype(numpy.float32)
        for i in range(10):
            _z[i] = (z[k + 1] * i + z[k] * (9 - i)) / 9.
        y = decoder([_z])[0]

        for i in range(10):
            img_path = 'morph.{}.{}.png'.format(k, i)
            # Image.fromarray(numpy.uint8(y[i] * 255.0))).save(img_path)
            Image.fromarray(numpy.transpose(numpy.uint8(y[i] * 255.0), (1, 0))).save(img_path)


if __name__ == '__main__':
    cli()
