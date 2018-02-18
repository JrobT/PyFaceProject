#!/usr/bin/env python3

"""A demonstration of a Convolutional Neural Network."""

# Import packages.
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# My imports.
import extraction_model as exmodel
import evaluation_model as evmodel
from sort_database.utils import EMOTIONS_5, EMOTIONS_8

# Required methods.
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Imports for Keras.
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU


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

tdata, tlabels, pdata, plabels = exmodel.get_sets_as_images(EMOTIONS_8)
tdata1, tlabels1, pdata1, plabels1 = exmodel.get_sets_as_images(EMOTIONS_5)

# Convert to readable arrays.
X_train = exmodel.convert_numpy(tdata)
y_train = exmodel.convert_numpy(tlabels)
X_test = exmodel.convert_numpy(pdata)
y_test = exmodel.convert_numpy(plabels)
X_train1 = exmodel.convert_numpy(tdata1)
y_train1 = exmodel.convert_numpy(tlabels1)
X_test1 = exmodel.convert_numpy(pdata1)
y_test1 = exmodel.convert_numpy(plabels1)
print('\nTraining data shape for 8 : ', X_train.shape, y_train.shape)
print('Testing data shape for 8 : ', X_test.shape, y_test.shape)
print('Training data shape for 5 : ', X_train1.shape, y_train1.shape)
print('Testing data shape for 5 : ', X_test1.shape, y_test1.shape)

# Find the unique numbers from the train labels.
classes = np.unique(y_train)
nClasses = len(classes)
classes1 = np.unique(y_train1)
nClasses1 = len(classes1)
print('\nTotal number of outputs for 8 : ', nClasses)
print('Output classes for 8 : ', classes)
print('Total number of outputs for 5 : ', nClasses1)
print('Output classes for 5 : ', classes1)

# Convert each 380x380 image into a 380x380x1 matrix.
X_train = X_train.reshape(-1, 380, 380, 1)
X_test = X_test.reshape(-1, 380, 380, 1)
X_train1 = X_train1.reshape(-1, 380, 380, 1)
X_test1 = X_test1.reshape(-1, 380, 380, 1)
print('\nTraining data shape for 8 : ', X_train.shape)
print('Testing data shape for 8 : ', X_test.shape)
print('Training data shape for 5 : ', X_train1.shape)
print('Testing data shape for 5 : ', X_test1.shape)

# Convert from int8 to float32 format, and scale the pixels.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.
X_train1 = X_train1.astype('float32')
X_test1 = X_test1.astype('float32')
X_train1 = X_train1 / 255.
X_test1 = X_test1 / 255.

# Change the labels from categorical to one-hot encoding.
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
y_train_one_hot1 = to_categorical(y_train1)
y_test_one_hot1 = to_categorical(y_test1)
print('\nOriginal label for 8 : ', y_train[0])
print('After conversion to one-hot for 8 : ', y_train_one_hot[0])
print('Original label for 5 : ', y_train1[0])
print('After conversion to one-hot for 5 : ', y_train_one_hot1[0])

# 80/20 split.
X_train, X_valid, train_lbl, valid_lbl = train_test_split(X_train,
                                                          y_train_one_hot,
                                                          test_size=0.2,
                                                          random_state=13)
X_train1, X_valid1, train_lbl1, valid_lbl1 = train_test_split(X_train1,
                                                              y_train_one_hot1,
                                                              test_size=0.2,
                                                              random_state=13)
print(X_train.shape, X_valid.shape, train_lbl.shape, valid_lbl.shape)
print(X_train1.shape, X_valid1.shape, train_lbl1.shape, valid_lbl1.shape)

batch_size = 64
epochs = 50
num_classes = 8
num_classes1 = 5

cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(380, 380, 1),
                     padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D((2, 2), padding='same'))
cnn_model.add(Conv2D(64, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Conv2D(128, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(Dense(num_classes, activation='softmax'))

cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
with open('results/cnn_summary_8', "w") as text_file:
    print(cnn_model.summary())

cnn_train = cnn_model.fit(X_train, train_lbl, batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_valid, valid_lbl))
y_pred = cnn_train.predict(X_test)

test_eval = cnn_model.evaluate(X_test, y_test_one_hot, verbose=0)
# Output the results.
with open('results/cnn_acc_8', "w") as text_file:
    print('Test loss : ', test_eval[0])
    print('Test accuracy : ', test_eval[1])

name = "cnn8"
evmodel.report(y_test, y_pred, nClasses, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), False, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), True, name)
evmodel.plot_chart(cnn_train, name)
cnn_model.save('model_8.h5')

cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(380, 380, 1),
                     padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D((2, 2), padding='same'))
cnn_model.add(Conv2D(64, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Conv2D(128, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(Dense(num_classes1, activation='softmax'))

cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
with open('results/cnn_summary_5', "w") as text_file:
    print(cnn_model.summary())

cnn_train = cnn_model.fit(X_train1, train_lbl1, batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_valid1, valid_lbl1))
y_pred = cnn_train.predict(X_test1)

test_eval = cnn_model.evaluate(X_test1, y_test_one_hot1, verbose=0)
# Output the results.
with open('results/cnn_acc_5', "w") as text_file:
    print('Test loss : ', test_eval[0])
    print('Test accuracy : ', test_eval[1])

name = "cnn5"
evmodel.report(y_test1, y_pred, nClasses1, name)
evmodel.matrix(y_test1, y_pred, np.unique(y_train1), False, name)
evmodel.matrix(y_test1, y_pred, np.unique(y_train1), True, name)
evmodel.plot_chart(cnn_train, name)
cnn_model.save('model_5.h5')


cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(380, 380, 1),
                     padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D((2, 2), padding='same'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(64, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(128, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Dropout(0.4))
cnn_model.add(Flatten())
cnn_model.add(Dense(128))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(num_classes, activation='softmax'))

cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
with open('results/cnn_dropout_summary_8', "w") as text_file:
    print(cnn_model.summary())

cnn_train = cnn_model.fit(X_train, train_lbl, batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_valid, valid_lbl))

test_eval = cnn_model.evaluate(X_test, y_test_one_hot, verbose=1)
# Output the results.
with open('results/cnn_dropout_acc_8', "w") as text_file:
    print('Test loss : ', test_eval[0])
    print('Test accuracy : ', test_eval[1])

name = "cnn_dropout"
evmodel.report(y_test, y_pred, nClasses, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), False, name)
evmodel.matrix(y_test, y_pred, np.unique(y_train), True, name)
evmodel.plot_chart(cnn_train, name)
cnn_model.save("model_dropout_8.h5py")


cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(380, 380, 1),
                     padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D((2, 2), padding='same'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(64, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(128, (3, 3), padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Dropout(0.4))
cnn_model.add(Flatten())
cnn_model.add(Dense(128))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(num_classes1, activation='softmax'))

cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
with open('results/cnn_dropout_summary_5', "w") as text_file:
    print(cnn_model.summary())

cnn_train = cnn_model.fit(X_train1, train_lbl1, batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_valid1, valid_lbl1))

test_eval = cnn_model.evaluate(X_test1, y_test_one_hot1, verbose=1)
# Output the results.
with open('results/model_dropout_acc_5', "w") as text_file:
    print('Test loss : ', test_eval[0])
    print('Test accuracy : ', test_eval[1])

name = "cnn_dropout"
evmodel.report(y_test1, y_pred, nClasses1, name)
evmodel.matrix(y_test1, y_pred, np.unique(y_train1), False, name)
evmodel.matrix(y_test1, y_pred, np.unique(y_train1), True, name)
evmodel.plot_chart(cnn_train, name)
cnn_model.save("model_dropout_5.h5py")

# End the script.
end = time.clock()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("\n***> Time elapsed: {:0>2}:{:0>2}:{:05.2f}."
      .format(int(hours), int(minutes), seconds))
