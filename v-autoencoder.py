import os
import click
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Convolution2D, Dense, Flatten, Input, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras import objectives

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
        mu = Dense(M)(h)
        vr = Dense(M)(h)
        model = Model(input=x, output=[mu, vr])
        return model

    def sampling(args):
        mu, vr = args
        eps = K.random_normal(shape=(M,), mean=0., std=1.)
        return mu + eps * K.exp(vr / 2)

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
    mu, vr = Encoder()(x)
    z = Lambda(sampling)([mu, vr])
    y = Decoder()(z)
    model = Model(input=x, output=y)

    loss_z = K.mean(K.square(mu) - vr + K.exp(vr))
    loss_y = objectives.binary_crossentropy(x, y) * M

    def loss(y_true, y_pred):
        return loss_y + loss_z

    def _loss_y(a, b):
        return K.mean(loss_y)

    def _loss_z(a, b):
        return K.mean(loss_z)

    model.compile(loss=loss, optimizer='adam', metrics=[_loss_y, _loss_z])
    return model


@click.group()
def cli():
    pass


@cli.command()
@click.option('--nb_epoch', type=int, default=200)
def train(nb_epoch):
    # setup
    x_train, y_train, x_test, y_test = make_dataset()
    model = make_model()
    # model save configure
    save_path = 'weights.hdf5'
    cp = ModelCheckpoint(filepath=save_path, monitor='loss', verbose=2, save_best_only=True, save_weights_only=True)
    # if os.path.exists(save_path):
    #     model.load_weights(save_path)
    # training
    model.fit(x_train, x_train, batch_size=20, nb_epoch=nb_epoch, callbacks=[cp])


@cli.command()
def test():
    pass


if __name__ == '__main__':
    cli()
