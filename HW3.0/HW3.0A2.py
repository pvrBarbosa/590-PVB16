#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 07:11:26 2021

@author: pedrob
"""

from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000) # load only the top 10,000 most frequent words

# Vectorize train and test data
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers


# Create the model with 3 hidden layers containing 32 neurons each
# Using RELU activation function and L2 regularization
model = models.Sequential()
model.add(layers.Dense(32, activation='relu',
                       kernel_regularizer='l2',
                       input_shape=(10000,)))
model.add(layers.Dense(32, activation='relu',
                       kernel_regularizer='l2'))
model.add(layers.Dense(32, activation='relu',
                       kernel_regularizer='l2'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])


# creating a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train the model using 5 epochs and store the results
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', 
              metrics=['acc'])
history = model.fit(partial_x_train,
                    partial_y_train, 
                    epochs=5,
                    batch_size=512,
                    validation_data=(x_val, y_val))




import matplotlib.pyplot as plt

# Plot train and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict["acc"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()

# Plot train and validation accuracy
acc_values = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




