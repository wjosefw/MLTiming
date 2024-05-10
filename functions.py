
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
    """"Calculate integral using Simpsons' rule"""
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

def constant_fraction_discrimination(vector, fraction = 0.9, shift = 30, plot = True):
    corrected_signal = np.zeros_like(vector)
    for i in range(vector.shape[0]):
      inverted_signal = np.roll(-vector[i,:],shift)
      inverted_signal[0:shift] = 0.
      fraction_signal = fraction*vector[i,:]
      corrected_signal[i,:] = inverted_signal + fraction_signal
      if plot:
        plt.plot(corrected_signal[i,:])
    return corrected_signal


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def pulso(t,t0, tau_rise = 15, tau_drop = 150, NOISE = True):
  y = (1 - np.exp(-(t-t0)/tau_rise))*np.exp(-(t-t0)/tau_drop) 
  y[y<0.] = 0.
  if NOISE:
    noise = np.random.normal(scale = 0.01, size = len(t)-t0)
    smoothed_noise = gaussian_filter1d(noise, sigma = 10)
    noise2 = np.random.normal(scale = 1e-4, size = t0)
    y[t0:] = y[t0:] + smoothed_noise
    y[:t0] = y[:t0] + noise2
  y = y / np.max(y)
  return y


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def delay_pulse_pair(pulse_set, time_step, delay_steps = 32, NOISE = True):
  """"Function to delay a pair of two pulses a number of time points."""
  
  INPUT = np.zeros_like(pulse_set)
  REF = np.zeros((pulse_set.shape[0],), dtype = np.float32)

  NRD0 = np.random.randint(delay_steps, size = pulse_set.shape[0])
  NRD1 = np.random.randint(delay_steps, size = pulse_set.shape[0])

  for i in range(pulse_set.shape[0]):
    N0 = NRD0[i]
    INPUT[i,:,0] = np.roll(pulse_set[i,:,0],N0)

    N1 = NRD1[i]
    INPUT[i,:,1] = np.roll(pulse_set[i,:,1],N1)
    REF[i] = time_step*(N0-N1) 
  
    if NOISE:
      noise0 = np.random.normal(scale = 1e-3, size = N0)
      noise1 = np.random.normal(scale = 1e-3, size = N1)
      INPUT[i,0:N0,0] = noise0
      INPUT[i,0:N1,1] = noise1
    else:
      INPUT[i,0:N0,0] = pulse_set[i,:,0]
      INPUT[i,0:N1,1] = pulse_set[i,:,1]
  
  return INPUT, REF

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def delay_pulse_4_channels(pulse_set, time_step, delay_steps = 20, NOISE = True):
  """"Function to delay a pair of two pulses a number of time points.
   First I delay the two pulses the same amount on channel zero and two.
    And then a different amount between each other in channel one and three. 
    That way I can have two reference time differences. The time
    differences betweem channel zero-one and two-three."""

  INPUT = np.zeros((pulse_set.shape[0],pulse_set.shape[1], 4))
  REF_pulse1_delayed = np.zeros((pulse_set.shape[0],),dtype = np.float32)
  REF_pulse2_delayed = np.zeros((pulse_set.shape[0],),dtype = np.float32)


  NRD0 = np.random.randint(delay_steps, size = pulse_set.shape[0])
  NRD1 = np.random.randint(delay_steps, size = pulse_set.shape[0])
  NRD2 = np.random.randint(delay_steps, size = pulse_set.shape[0])

  for i in range(pulse_set.shape[0]):
    N0 = NRD0[i]
    INPUT[i,:,0] = np.roll(pulse_set[i,:,0],N0)
    INPUT[i,:,2] = np.roll(pulse_set[i,:,1],N0)

    N1 = NRD1[i]
    INPUT[i,:,1] = np.roll(pulse_set[i,:,0],N1)

    N2 = NRD2[i]
    INPUT[i,:,3] = np.roll(pulse_set[i,:,1],N2)


    REF_pulse1_delayed[i] = time_step*(N0-N1) 
    REF_pulse2_delayed[i] = time_step*(N0-N2)  

    if NOISE:
      noise00 = np.random.normal(scale = 0.01, size = pulse_set.shape[1])
      noise11 = np.random.normal(scale = 0.01, size = pulse_set.shape[1])
      noise22 = np.random.normal(scale = 0.01, size = pulse_set.shape[1])
      smoothed_noise_00 = gaussian_filter1d(noise00, sigma = 10)
      smoothed_noise_11 = gaussian_filter1d(noise11, sigma = 10)
      smoothed_noise_22 = gaussian_filter1d(noise22, sigma = 10)
      INPUT[i,0:N0,0] = smoothed_noise_00[0:N0]
      INPUT[i,0:N0,2] = smoothed_noise_00[0:N0]
      INPUT[i,0:N1,1] = smoothed_noise_11[0:N1]
      INPUT[i,0:N2,3] = smoothed_noise_22[0:N2]
    else:
      INPUT[i,0:N0,0] = pulse_set[i,0:N0,0]
      INPUT[i,0:N0,2] = pulse_set[i,0:N0,1]
      INPUT[i,0:N1,1] = pulse_set[i,0:N1,0]
      INPUT[i,0:N2,3] = pulse_set[i,0:N2,1]

  return INPUT, REF_pulse1_delayed, REF_pulse2_delayed    


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def get_mean_pulse_from_set(pulse_set, channel = 0):
    transforms = []
    
    for i in range(pulse_set.shape[0]):
        fourier_transform = np.fft.fft(pulse_set[i,:,channel])
        transforms.append(fourier_transform)
    
    transforms = np.array(transforms, dtype='object')
    sum_of_transf = np.sum(transforms, axis = 0) #sum all fourier transforms
    reconstructed_signal = np.fft.ifft(sum_of_transf) #inverse fourier transf.
    normalized_reconstructed_signal = reconstructed_signal/np.max(reconstructed_signal)
    mean_pulse = np.real(normalized_reconstructed_signal)
    
    return mean_pulse

    
   
