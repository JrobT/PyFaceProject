#!/usr/bin/env python3

"""A demonstration of a Convolutional Neural Network.

TODO : ADD DESCRIPTION
"""

# Import packages.
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# My imports.
import extraction_model as exmodel
import evaluation_model as evmodel
from constants import EMOTIONS_5

# Required methods.
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Imports for Keras.
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D


def demo_images(X_train, y_train, X_test, y_test):
    """Demo the first images and truths in dataset."""
    plt.figure(figsize=[5, 5])

    # Display the first image in training data.
    plt.subplot(121)
    plt.imshow(X_train[0, :, :], cmap='gray')
    plt.title("Ground Truth : {}".format(y_train[0]))

    # Display the first image in testing data.
    plt.subplot(122)
    plt.imshow(X_test[0, :, :], cmap='gray')
    plt.title("Ground Truth : {}".format(y_test[0]))

    plt.show()


print(__doc__)

# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\n{}: Beginning CNN test...\n".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate

tdata, tlabels, pdata, plabels = exmodel.get_sets_as_images(EMOTIONS_5)

# Convert to readable arrays.
X_train = exmodel.convert_numpy(tdata)
y_train = exmodel.convert_numpy(tlabels)
X_test = exmodel.convert_numpy(pdata)
y_test = exmodel.convert_numpy(plabels)
# print('Training data shape : ', X_train.shape, y_train.shape)
# print('Testing data shape : ', X_test.shape, y_test.shape)

# Find the unique numbers from the train labels.
classes = np.unique(y_train)
nClasses = len(classes)
# print('Total number of outputs : ', nClasses)
# print('Output classes : ', classes)

# Convert each 380x380 image into a 380x380x1 matrix.
X_train = X_train.reshape(-1, 380, 380, 1)
X_test = X_test.reshape(-1, 380, 380, 1)
# print('Training data shape : ', X_train.shape)
# print('Testing data shape : ', X_test.shape)

# Convert from int8 to float32 format, and scale the pixels.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.

# Change the labels from categorical to one-hot encoding.
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
# print('Original label : ', y_train[0])
# print('After conversion to one-hot : ', train_Y_one_hot[0])

# 80/20 split.
X_train, X_valid, train_lbl, valid_lbl = train_test_split(X_train,
                                                          y_train_one_hot,
                                                          test_size=0.2,
                                                          random_state=13)
# print(X_train.shape, X_valid.shape, train_lbl.shape, valid_lbl.shape)

batch_size = 64
epochs = 20
num_classes = 5

cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                     input_shape=(380, 380, 1), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D((2, 2), padding='same'))
cnn_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='linear'))
cnn_model.add(Activation('relu'))
cnn_model.add(Dense(num_classes, activation='softmax'))

cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
with open('results/cnn_summary', "w") as text_file:
    print(cnn_model.summary())

cnn_train = cnn_model.fit(X_train, train_lbl, batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_valid, valid_lbl))
y_pred = cnn_train.predict(X_test)

test_eval = cnn_model.evaluate(X_test, y_test_one_hot, verbose=0)
# Output the results.
with open('results/cnn_acc', "w") as text_file:
    print('Test loss : ', test_eval[0])
    print('Test accuracy : ', test_eval[1])

name = "cnn"
evmodel.report(y_test, y_pred, nClasses, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), False, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), True, name)
evmodel.plot_chart(cnn_train, name)

cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                     input_shape=(380, 380, 1), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D((2, 2), padding='same'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Dropout(0.4))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='linear'))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(num_classes, activation='softmax'))

cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
with open('results/cnn_dropout_summary', "w") as text_file:
    print(cnn_model.summary())

cnn_train = cnn_model.fit(X_train, train_lbl, batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_valid, valid_lbl))

test_eval = cnn_model.evaluate(X_test, y_test_one_hot, verbose=1)
# Output the results.
with open('results/cnn_dropout_acc', "w") as text_file:
    print('Test loss : ', test_eval[0])
    print('Test accuracy : ', test_eval[1])

name = "cnn_dropout"
evmodel.report(y_test, y_pred, nClasses, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), False, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), True, name)
evmodel.plot_chart(cnn_train, name)

# cnn_model.save("cnn_model_dropout.h5py")

# End the script.
end = time.clock()
print("""\n{}: Program end. Time elapsed:
      {:.5f}.""".format(script_name, end - start))
