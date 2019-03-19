import sys

#   Recognition of handwritten numbers - TEST №1
#   see more: http://yann.lecun.com/exdb/mnist/
#   
#   using libraries:
#       numpy
#       TensorFlow
#       Keras

import numpy # different math; arrays; matrix; very productive
import tensorflow as tf # machine learning
from keras.models import Sequential # linear stack of layers
from keras.layers import Dense # neuro: activation func etc; FEEDFORWARD NEURALNETWWORK
from keras.utils import np_utils # different utils

# stochastic optimization setting: current value for the data repeatability
numpy.random.seed(23)
# tensorflow mnist set of numbers data
mnist = tf.keras.datasets.mnist


def main():
    # DATA PREPROCESSING

    (x_training_set, y_training_set), (x_test_set, y_test_set) = mnist.load_data()
    x_training_set = x_training_set.reshape(60000, 784) # image's sizing reformation

    x_training_set = x_training_set.astype('float32') #data normalize
    x_training_set /= 255

    y_training_set = np_utils.to_categorical(y_training_set, 10) # transform neural output label into category - 10 digit
    # 0 --> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 1 --> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # 2 --> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # 3 --> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # 4 --> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # 5 --> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # 6 --> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    # 7 --> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # 8 --> [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # 9 --> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # NEURAL NETWORK 

    neural = Sequential() # creating feedforward neural

    # model type: Dense

    # input layer with 500 neuron each contains 784 input's; 
    # distribution: normal(Gaussian)
    # activation func: rectifier(ReLU); see more: https://en.wikipedia.org/wiki/Rectifier_(neural_networks) 
    neural.add(Dense(500, input_dim=784, kernel_initializer='normal', activation='relu')) 

    # output layer with 10 neuron
    # distribution: normal(Gaussian)
    # activation func: softmax; see more: https://en.wikipedia.org/wiki/Softmax_function
    neural.add(Dense(10, kernel_initializer='normal', activation='softmax'))

    # compiling neural
    # teaching method: SGD; see more: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    # measure of error: categorical_crossentropy ???
    # metrics optimization: accuracy
    neural.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    print(neural.summary())

if __name__ == '__main__':
    main()
    print('test#1 done...')
else:
    pass