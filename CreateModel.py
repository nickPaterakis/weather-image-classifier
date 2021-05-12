import cv2
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from LoadData import X_train, Y_train, X_test, Y_test, X_val, Y_val, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
import matplotlib.pyplot as plt
import numpy as np


# Model configuration
batch_size = 32
loss_function = sparse_categorical_crossentropy
no_epochs = 25
optimizer = SGD(lr=0.001, momentum=0.9)
verbosity = 1
input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# class 0 cloudy
# class 1 rain
# class 2 shine
# class 3 sunrise
classes = ["Cloudy", "Rain", "Shine", "Sunrise"]

print("Data creation completed, we have", X_train.shape[0], "training paradigms of size",
      X_train.shape[1], "x", X_train.shape[2], ".\n", X_test.shape[0],
      "test paradigms of the same size and", Y_test.shape[0],
      "test paradigms also of the same size")

class_to_demonstrate = 0
while (sum(Y_train == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(Y_train == class_to_demonstrate)

    # create new plot window
    plt.figure()

    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(X_train[tmp_idxs_to_use[0][0]], cv2.COLOR_BGR2RGB))
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(X_train[tmp_idxs_to_use[0][1]], cv2.COLOR_BGR2RGB))
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(X_train[tmp_idxs_to_use[0][2]], cv2.COLOR_BGR2RGB))
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(X_train[tmp_idxs_to_use[0][3]], cv2.COLOR_BGR2RGB))
    tmp_title = str(classes[class_to_demonstrate]) + ' images'
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1

# Parse numbers as floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')

# Normalize data
X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

print("CNN topology setup completed")

# print model summary
model.summary()

# create data generator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# prepare iterator
it_train = datagen.flow(X_train, Y_train, batch_size=batch_size)

steps = int(X_train.shape[0] / 32)
history = model.fit_generator(
    it_train,
    steps_per_epoch=steps,
    epochs=no_epochs,
    validation_data=(X_val, Y_val),
    verbose=verbosity)

print("Model fitting is completed")

model_name = 'CNN_model.h5'
model.save(model_name)

print("Model is saved")

# Generate generalization metrics
score = model.evaluate(X_test, Y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()