
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def fix_imbalances(vector, value=0, window = 0.4):
    """ Function to remove random elements
    of an array centered around a specific
    value """
    top_threshold = value + window
    lower_threshold = value - window    
    index = np.where((vector > lower_threshold) & (vector < top_threshold))[0]
    np.random.shuffle(index)
    index_to_delete = index[:int(0.5*index.shape[0])]
    return index_to_delete


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

from scipy.optimize import curve_fit
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def momentos(vector):
  Nev,Nt,Nc = np.shape(vector)    #Nev: Núm eventos, Nt: Núm puntos temporales, Nc: Número canales
  t = np.reshape(np.linspace(0,Nt, Nt)/float(Nt),(1,-1,1))#Normalized array of time
  MOMENT = np.zeros((Nev,0,Nc))

  for i in range(4): #Number of moments used
    W = t**(i+1)
    W = np.tile(W,(Nev,1,Nc))
    MM = np.sum(vector*W,axis=1,keepdims=True)
    MOMENT = np.append(MOMENT,MM,axis=1)

    W = np.exp(-(i)*t)
    W = np.tile(W,(Nev,1,Nc))
    MM = np.sum(vector*W,axis=1,keepdims=True)
    MOMENT = np.append(MOMENT,MM,axis=1)

    W = np.exp(-(t**i))
    W = np.tile(W,(Nev,1,Nc))
    MM = np.sum(vector*W,axis=1,keepdims=True)
    MOMENT = np.append(MOMENT,MM,axis=1)

  return MOMENT

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

@njit
def normalize_by_max(array_pulsos):
  y = np.zeros_like(array_pulsos)
  for i in range(array_pulsos.shape[0]):
    y[i,:,0] =  array_pulsos[i,:,0] / np.max(array_pulsos[i,:,0])
    y[i,:,1] =  array_pulsos[i,:,1] / np.max(array_pulsos[i,:,1])
  return y

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def simpsons_rule_array(y, h):
    array = np.zeros(y.shape[0])
    n = y.shape[1]

    for i in range(y.shape[0]):
      integral = y[i,0] + y[i,-1]

      for j in range(1, n, 2):
          integral += 4 * y[i,j]

      for j in range(2, n - 1, 2):
          integral += 2 * y[i,j]

      integral *= h / 3
      array[i] = integral

    return array

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

from scipy import signal

def get_correlation(ref_pulse, pulse_set, channel = 0):
  """Function to calculate the correlation of a
  reference pulse to every pulse in a set"""
  y1 = ref_pulse
  n = len(y1)
  correlation = []
  for i in range(pulse_set.shape[0]):
    y2 = pulse_set[i,:,channel]
    corr = signal.correlate(y2, y1, mode = 'same')
    correlation.append(corr[n//2]) #Correlation at delay zero
  correlation = np.array(correlation)
  return correlation

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def get_closest(ref_pulse, pulse_set, channel = 0):
  """Function to calculate the most similar
  pulse in a set to a reference one"""
  y1 = ref_pulse
  mse = []
  for i in range(pulse_set.shape[0]):
    y2 = pulse_set[i,:,channel]
    mse.append(np.mean((y1-y2)**2))
  mse = np.array(mse)
  sorted_indices = np.argsort(mse)
  index_of_closest = sorted_indices[1]
  return index_of_closest

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_set(og_set, channel = 0):
  new_set = np.zeros_like(og_set)
  for i in range(og_set.shape[0]):
    closest = get_closest(og_set[i,:,channel], og_set, channel = channel)
    new_set[i,:,0] = og_set[i,:,channel]
    new_set[i,:,1] = og_set[closest,:,channel]
  return new_set


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def weights_definition(NM, Npoints):
  t = np.linspace(0,Npoints, Npoints)/float(Npoints) #Normalized array of time
  Weights = np.zeros((Npoints,NM))
  NMW = int(NM/3) #Number of Moments per weight

  for i in range(NMW):
    Weights[:,i] = t**(i+1)

  for i in range(NMW):
    Weights[:,i + NMW] = np.exp(-t**(i))

  for i in range(NMW):
    Weights[:,i + 2*(NMW)] = np.exp(-(i)*t)

  return Weights

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def constant_fraction_discrimination(vector, fraction = 0.9, shift = 30):
    corrected_signal = np.zeros_like(vector)
    for i in range(vector.shape[0]):
      inverted_signal = np.roll(-vector[i,:],shift)
      inverted_signal[0:shift] = 0.
      fraction_signal = fraction*vector[i,:]
      corrected_signal[i,:] = inverted_signal + fraction_signal
      plt.plot(corrected_signal[i,:])
    return corrected_signal



