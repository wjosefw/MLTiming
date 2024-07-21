
import numpy as np
import tensorflow as tf
tf.version.VERSION

from tensorflow.keras.layers import Input, Layer, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, concatenate
from tensorflow.keras import initializers
from keras.regularizers import l2
from tensorflow.keras.initializers import Constant

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def model_definition_manual_weight_initialization(N_time_points, kernel_weights, NM = 12, NN = 32, STD_INIT = 1):

  inputA = Input(shape=(N_time_points,))
  inputB = Input(shape=(N_time_points,))

  x1 = Dense(NM, activation="relu", kernel_initializer= Constant(kernel_weights), bias_initializer='zeros')
  #x1 = Dense(NM, activation = "relu", kernel_initializer=initializers.RandomNormal(mean = 0.0, stddev = STD_INIT), kernel_regularizer = l2(0.01))
  x2 = Dense(NN, activation = "relu", kernel_initializer = initializers.RandomNormal(mean = 0.0, stddev = STD_INIT), kernel_regularizer = l2(0.01))
  x3 = Dense(NN, activation = "relu", kernel_initializer = initializers.RandomNormal(mean = 0.0, stddev = STD_INIT), kernel_regularizer = l2(0.01))
  x4 = Dense(NN, activation = "relu", kernel_initializer = initializers.RandomNormal(mean = 0.0, stddev = STD_INIT), kernel_regularizer = l2(0.01))
  x5 = Dense(1, activation = "linear")

  op1A = x1(inputA)
  op2A = x2(op1A)
  op3A = x3(op2A)
  op4A = x4(op3A)
  outA = x5(op4A)

  op1B = x1(inputB)
  op2B = x2(op1B)
  op3B = x3(op2B)
  op4B = x4(op3B)
  outB = x5(op4B)

  model = Model(inputs = [inputA, inputB], outputs = outA-outB)
  return model 


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def model_definition(NM = 12, NN = 128, STD_INIT = 1):

  inputA = Input(shape=(NM,))
  inputB = Input(shape=(NM,))

  x1 = Dense(NN, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev = STD_INIT), kernel_regularizer=l2(0.01))
  x2 = Dense(NN, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev = STD_INIT), kernel_regularizer=l2(0.01))
  x3 = Dense(NN, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev = STD_INIT), kernel_regularizer=l2(0.01))
  x4 = Dense(1, activation="linear")

  op1A = x1(inputA)
  op2A = x2(op1A)
  op3A = x3(op2A)
  outA = x4(op3A)

  op1B = x1(inputB)
  op2B = x2(op1B)
  op3B = x3(op2B)
  outB = x4(op3B)

  model = Model(inputs = [inputA, inputB], outputs = outA - outB)
  return model

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def model_definition_HEinitialization(NM = 12, NN = 128):

  inputA = Input(shape=(NM,))
  inputB = Input(shape=(NM,))

  x1 = Dense(NN, activation="relu", kernel_initializer = initializers.HeNormal(), kernel_regularizer = l2(0.01))
  x2 = Dense(NN, activation="relu", kernel_initializer = initializers.HeNormal(), kernel_regularizer = l2(0.01))
  x3 = Dense(NN, activation="relu", kernel_initializer = initializers.HeNormal(), kernel_regularizer = l2(0.01))
  x4 = Dense(1, activation="linear")

  op1A = x1(inputA)
  op2A = x2(op1A)
  op3A = x3(op2A)
  outA = x4(op3A)

  op1B = x1(inputB)
  op2B = x2(op1B)
  op3B = x3(op2B)
  outB = x4(op3B)

  model = Model(inputs=[inputA, inputB], outputs = outA-outB)
  return model


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def model_definition_convolutional(N_points):

  input_shape = (2,N_points,1)


  model = Sequential()
  model.add(Conv2D(filters = 16, kernel_size=(2, 2), activation='relu', padding = 'same', input_shape=input_shape)) #ouput_dim = (2, N_points, 4)
  model.add(MaxPooling2D((2, 2))) # Halves the dimensions
  model.add(Conv2D(filters = 32, kernel_size=(2, 2), activation='relu', padding = 'same')) #ouput_dim = (1, N_points/2, 4)
  model.add(MaxPooling2D((1, 2)))
  model.add(Flatten())
  model.add(Dense(4, activation='relu'))
  model.add(Dense(1, activation='linear'))
  return model


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def model_definition_four_inputs(NM = 12, NN = 32, STD_INIT = 1):

  inputA = Input(shape=(NM,))
  inputB = Input(shape=(NM,))
  inputC = Input(shape=(NM,))
  inputD = Input(shape=(NM,))

  x1 = Dense(NN, activation="relu", kernel_initializer = initializers.RandomNormal(mean = 0.0, stddev = STD_INIT) , kernel_regularizer = l2(0.01))
  x2 = Dense(NN, activation="relu", kernel_initializer = initializers.RandomNormal(mean = 0.0, stddev = STD_INIT) , kernel_regularizer = l2(0.01))
  x3 = Dense(NN, activation="relu", kernel_initializer = initializers.RandomNormal(mean = 0.0, stddev = STD_INIT) , kernel_regularizer = l2(0.01))
  x4 = Dense(1, activation="relu")

  op1A = x1(inputA)
  op2A = x2(op1A)
  op3A = x3(op2A)
  outA = x4(op3A)

  op1B = x1(inputB)
  op2B = x2(op1B)
  op3B = x3(op2B)
  outB = x4(op3B)

  op1C = x1(inputC)
  op2C = x2(op1C)
  op3C = x3(op2C)
  outC = x4(op3C)

  op1D = x1(inputD)
  op2D = x2(op1D)
  op3D = x3(op2D)
  outD = x4(op3D)

  final_output = concatenate([outA, outB, outC, outD], axis=-1)
  model = Model(inputs=[inputA, inputB, inputC, inputD], outputs=final_output)
  return model