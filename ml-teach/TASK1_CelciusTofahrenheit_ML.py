from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

# training set
CELSIUS_DATA    = np.array([-128, -64, -32, 0,  8,  16, 32, 64,  128], dtype=float) # start data pull
FAHRENHEIT_DATA = np.array([-198, -83, -25, 32, 46, 61, 90, 147, 262], dtype=float) # desired result

def TASK1_Run():
    TASK1_Model_Prediction(TASK1_CreateAndTeachModel())

def TASK1_CreateAndTeachModel():

    # creating Network layer
    net = tf.keras.layers.Dense(units=1, input_shape=[1])

    # creating Model
    model = tf.keras.Sequential([net]) # layers in order of distribution

    # model compilation
    # loss function - mean squared
    # optimization function - Adam algorithm
    # learning rate - to little value: speed drops; to mach value: accuracy drops
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.15))

    # model training
    model.fit(CELSIUS_DATA, FAHRENHEIT_DATA, epochs=1000, verbose=False)
    print('Model training done...\n')
    return model

def TASK1_Model_Prediction(model):
    value = float(input('Enter the celsius degree: '))
    print(value, 'C degree = ', model.predict([value]), 'F degree')