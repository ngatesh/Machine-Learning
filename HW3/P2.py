from keras.models import Sequential
from keras.layers import Dense
from numpy.random import rand, randn
import numpy as np


def def_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def def_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model


def def_gan(gen, disc):
    model = Sequential()
    model.add(gen)
    model.add(disc)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_real_samples(n):
    u = 0
    s = np.sqrt(0.2)

    X1 = 2*rand(n) - 1
    X2 = 1/(s*np.sqrt(2*np.pi)) * np.exp(-0.5 * (X1-u)**2 / s**2)

    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))

    y = np.ones((n, 1))
    return X, y


def generate_fake_samples(n):
    X1 = 2*rand(n) - 1
    X2 = 2*rand(n) - 1

    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))

    y = np.zeros((n, 1))
    return X, y


def generate_latent_points(latent_dim, n):
    x_input = randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input


def train(d_model, gan_model, latent_dim, n_epochs=100, n_batch=128):
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(half_batch)

        d_model.trainable = True
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)

        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))

        d_model.trainable = False
        acc = gan_model.train_on_batch(x_gan, y_gan)
        print(f'Accuracy: {acc[1]:0.4f}')


discriminator = def_discriminator()
generator = def_generator(1)
gan = def_gan(generator, discriminator)

train(discriminator, gan, 1)

X, y_real = generate_real_samples(30)

