# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:44:22 2020

@author: Anonymous
"""
"""Basic Image Classification
Now, we will build a neural network model to solve a basic image classification problem. We will accomplish this with the help of following tasks in the project:

Understand the problem statement
Understand the dataset
Encode the labels
Understand neural networks
Preprocess image examples
Create a neural network model
Train the model to fit the dataset
Evaluate the model
Visualize the predictions"""
import numpy as np 
import  tensorflow as tf 
#tf.logging.set_verbosity(tf.logging.ERROR)
print('Using  Tensorflow version',tf.__version__)
# The Dataset (import MNIST)
from tensorflow .keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()
# shape of imported arrays
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)

#plot an Image Example
#train set example 1
from matplotlib import pyplot as plt 
plt.imshow(x_train[0], cmap='binary')
plt.show()
# train set example 2  
plt.imshow(x_train[1], cmap='binary')
plt.show()

# test example 1
plt.imshow(x_test[0], cmap='binary')
plt.show()

# display labels
y_train[0]
y_test[0]

# One Hot Encoding 

# After this encoding, every label will be converted to a list with 10 
# elements and the element at index to the coressponding class will be
# set to 1, rest will be set to 0

"""eg 5 : [0,0,0,0,0,1,0,0,0,0]
      7 : [0,0,0,0,0,0,0,1,0,0]
      1 : [0,1,0,0,0,0,0,0,0,0]
"""

# encoding labels
#import keras_utils

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)

# validated Shapes
print('y_train_encoded shape: ', y_train_encoded.shape)
print('y_test_encoded shape: ', y_test_encoded.shape)

# Display Encoded Labels
y_train_encoded[0]

# Neural Networks
#Linear Equations
# pre processing examples 
# Unrolling N-dimensional Arrays to Vectors


x_train_reshaped  = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test,  (10000, 784))
print('x_train_reshaped shape: ', x_train_reshaped.shape)
print('x_test_reshaped shape: ', x_test_reshaped.shape)

# Display Pixel Values
print(set(x_train_reshaped[0]))

# Data Normalisation

x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10
x_train_norm = (x_train_reshaped- x_mean)/(x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean)/(x_std + epsilon)

# Display Normalised Pixel Values
print(set(x_train_norm[0]))

# Creating the Model

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(120, activation = 'relu', input_shape= (784,)),
    Dense(120, activation= 'relu'),
    Dense(10,  activation='softmax')
    ])
#Compiling the Model

model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
    )
model.summary()

# Training the Model

model.fit(x_train_norm, y_train_encoded, epochs=3)

#Evaluating the Model

loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy: ', accuracy*100)

# Predicitions on Test Set
preds = model.predict(x_test_norm)
print('Shape of preds: ',preds.shape)

# Plotting the Results 

plt.figure(figsize = (12, 12))
start_index = 0

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index])
    gt = y_test[start_index+i]
    
    col = 'g'
    if pred !=gt:
        col = 'r'
        
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i,pred,gt))
    plt.imshow(x_test[start_index+i], cmap = 'binary')
plt.show()

plt.plot(preds[0])
plt.show()




