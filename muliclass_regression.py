'''
Drew Meyer
7/27/23
Multiclass Regression model for recognizing handwritten digits on the MNIST 
'''

import tensorflow as tf # import tensorflow library
from tensorflow.keras import Sequential # import sequential model
from tensorflow.keras.layers import Dense # import forward prop layer architecture
from tensorflow.keras.losses import SparseCategoricalCrossentropy # import crossentropy loss function

# define model architecture
model = Sequential([
  Dense(units=25, activation='relu'), # define first layer with 25 neurons and relu activation function
  Dense(units=15, activation='relu'), # define second layer with 15 neurons and relu activation function
  Dense(units=10, activation='linear') # define third and final layer
]) 

# specify loss and cost function
model.compile(loss = SparseCategoricalCrossentropy(from_logits=True) )
# from_logits=True improves rounding errors for numnerical accuracy

# train on data to minimize cost function
model.fit(X, Y, epochs=100)

# predict class for given digit
logits = model(X)
f_x = tf.nn.softmax(logits)
