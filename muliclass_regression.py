'''
Drew Meyer
7/27/23
Multiclass Regression model for recognizing handwritten digits on the MNIST 
'''

import tensorflow as tf # import tensorflow library
from tensorflow.keras import Sequential # import sequential model
from tensorflow.keras.layers import Dense # import forward prop layer architecture
from tensorflow.keras.losses import SparseCategoricalCrossentropy # import crossentropy loss function

model = Sequential([
  Dense(units=25, activation='relu'), # define first layer with 25 neurons and relu activation function
  Dense(units=15, activation='relu'), # define second layer with 15 neurons and relu activation function
  Dense(units=10, activation='linear') # define third and final layer
]) 

model.compile(loss = SparseCategoricalCrossentropy(from_logits=True) ) # specify loss and cost function
# from_logits=True improves rounding errors for numnerical accuracy

model.fit(X, Y, epochs=100) # train on data to minimize cost function

logits = model(X)
f_x = tf.nn.softmax(logits) # predict class for given digit
