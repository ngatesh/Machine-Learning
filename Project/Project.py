from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from matplotlib import pyplot as plt

# Initialize Model
model = Sequential()

# Add the first convolutional layer, 2x2 pooling, and 20% dropout.
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Add the second convolutional layer, 2x2 pooling, and 20% dropout.
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Add the third convolutional layer, 2x2 pooling, and 20% dropout.
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Flatten the model into a fully-connected network, then converge to two outputs (dog/cat).
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Print model summary, so we can see what it looks like.
model.summary()

# Compile the network to use a binary cross entropy loss function.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Scale image pixel values to work better with the optimizer's (adam) default learning rate.
# dataGen = ImageDataGenerator(rescale=1./255, shear_range=0.25, zoom_range=0.25, horizontal_flip=True)
dataGen = ImageDataGenerator(rescale=1./255)

# Initialize image data streams. Model will be trained in batches of 32 images per step, 64x64 px images.
trainingData = dataGen.flow_from_directory("donuts-v-muffins/train", target_size=(64, 64), batch_size=32)
testingData = dataGen.flow_from_directory("donuts-v-muffins/test", target_size=(64, 64), batch_size=32)

# Train the network. 20,000 training images / 32 = 625, so 1 epoch = full training set. Train 10 times.
# 5,000 validation images / 32 = 156.
history = model.fit(trainingData, steps_per_epoch=36, epochs=30, validation_data=testingData, validation_steps=9)

# Plot Results
plt.title('Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')
plt.legend(['Training', 'Validation'])
plt.ylim(0.5, 1.0)

plt.show()
