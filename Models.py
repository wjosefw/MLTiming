
import numpy as np
import tensorflow as tf
tf.version.VERSION

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from keras.regularizers import l2


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def model_definition_manual_weight_initialization(N_time_points, NM, weights):

  inputA = Input(shape=(N_time_points,))
  inputB = Input(shape=(N_time_points,))

  x1 = Dense(NM, activation="relu", weights=[weights, np.zeros(NM)])
  #x1 = Dense(NM, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1.5), kernel_regularizer=l2(0.01))
  x2 = Dense(4, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1.5), kernel_regularizer=l2(0.01))
  x3 = Dense(4, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1.5), kernel_regularizer=l2(0.01))
  x4 = Dense(4, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1.5), kernel_regularizer=l2(0.01))
  x5 = Dense(1, activation="linear")

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

  model = Model(inputs=[inputA, inputB], outputs=outA-outB)
  return model


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def model_definition(NM, NN = 128, STD_INIT = 1):

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

  model = Model(inputs=[inputA, inputB], outputs=outA-outB)
  return model