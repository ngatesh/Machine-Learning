from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

trainingData = ImageDataGenerator().flow_from_directory("C:\\DogsCats\\dogs-vs-cats\\train", target_size=(64, 64))
testingData = ImageDataGenerator().flow_from_directory("C:\\DogsCats\\dogs-vs-cats\\test", target_size=(64, 64))

model.fit(trainingData, steps_per_epoch=500, epochs=40, validation_data=testingData, validation_steps=5000)

model.save('classifier.h5')