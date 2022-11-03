from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from numpy.random import rand, randn
import numpy as np
import matplotlib.pyplot as plt


def def_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
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
    X1 = 20*rand(n) - 10
    X2 = 20*rand(n) - 10

    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))

    y = np.zeros((n, 1))
    return X, y


def generate_latent_points(n):
    x_input = randn(n)
    x_input = x_input.reshape(n)
    return x_input


def trainGAN(d_model, gan_model, n_epochs=5000, n_batch=128):
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake = generator(generate_latent_points(half_batch))
        y_fake = np.zeros((half_batch, 1))

        x_train = np.vstack((x_real, x_fake))
        y_train = np.vstack((y_real, y_fake))

        d_model.trainable = True
        accD = d_model.train_on_batch(x_train, y_train)

        x_gan = generate_latent_points(n_batch)
        y_gan = np.ones((n_batch, 1))

        d_model.trainable = False
        accG = gan_model.train_on_batch(x_gan, y_gan)

        if i % 100 == 0:
            print(f'Epoch: {i}\tDisc_Acc: {accD[1]:0.4f}\tGAN_Acc: {accG[1]:0.4f}')


def trainDiscriminator(d_model, n_epochs=1000, n_batch=512):
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(half_batch)
        x_train = np.vstack((x_real, x_fake))
        y_train = np.vstack((y_real, y_fake))

        d_model.trainable = True
        acc = d_model.train_on_batch(x_train, y_train)

        if i % 100 == 0:
            print(f'Disc:\tEpoch: {i}\tAccuracy: {acc[1]:0.4f}')
            if acc[1] > 0.995:
                break


discriminator = def_discriminator()
generator = def_generator(1)
gan = def_gan(generator, discriminator)

trainDiscriminator(discriminator)
trainGAN(discriminator, gan)

X_real, _ = generate_real_samples(100)
X_fake, _ = generate_fake_samples(100)
X_test = generator(generate_latent_points(100))

plt.scatter(X_real[:, 0], X_real[:, 1])
# plt.scatter(X_fake[:, 0], X_fake[:, 1])
plt.scatter(X_test[:, 0], X_test[:, 1])
plt.show()


