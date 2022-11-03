from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from numpy.random import rand, randn
import numpy as np
import matplotlib.pyplot as plt

# Author: Nathaniel Gatesh

# Define the discriminator.
def def_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# Define the generator.
def def_generator(n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=1))
    model.add(Dense(n_outputs, activation='linear'))
    return model


# Put the generator and discriminator to form the Generative Adversarial Network
def def_gan(gen, disc):
    model = Sequential()
    model.add(gen)
    model.add(disc)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Generates a set of random numbers in a Gaussian Distribution.
# Returns X=[x, G(x)], Y = ones(n,1)
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


# Generates random points to feed into the generator.
def generate_latent_points(n):
    x_input = randn(n)
    x_input = x_input.reshape(n)
    return x_input


# Train the GAN.
def trainGAN(d_model, gan_model, n_epochs=20000, n_batch=128):
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)      # Real Gaussian curve data.
        x_fake = generator(generate_latent_points(half_batch))  # Fake data from generator.
        y_fake = np.zeros((half_batch, 1))                      # Zeros tell optimizer to recognize fake data.

        x_train = np.vstack((x_real, x_fake))   # Combine inputs into one batch.
        y_train = np.vstack((y_real, y_fake))   # Combine correct outputs into one batch.

        # Train the discriminator.
        d_model.trainable = True
        accD = d_model.train_on_batch(x_train, y_train)

        # Get more data from generator.
        x_gan = generate_latent_points(n_batch)
        y_gan = np.ones((n_batch, 1))

        # Train the generator through the GAN, attempting to trick discriminator into outputting 1's.
        d_model.trainable = False
        accG = gan_model.train_on_batch(x_gan, y_gan)

        # Print accuracy stats.
        if i % 100 == 0:
            print(f'Epoch: {i}\tDisc_Acc: {accD[1]:0.4f}\tGAN_Acc: {accG[1]:0.4f}')


# Initialize the discriminator, generator, and GAN.
discriminator = def_discriminator()
generator = def_generator()
gan = def_gan(generator, discriminator)

# Train the GAN.
trainGAN(discriminator, gan)

# Plot real vs. generated data.
X_real, _ = generate_real_samples(100)
X_gen = generator(generate_latent_points(100))

plt.scatter(X_real[:, 0], X_real[:, 1])
plt.scatter(X_gen[:, 0], X_gen[:, 1])
plt.title("GAN-Produced Gaussian Curve")
plt.legend(['Real', 'Generated'])
plt.show()
