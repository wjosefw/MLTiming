
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def fix_imbalances(vector, value = 0, window = 0.4):
    """
    Remove random elements of an array centered around a specific value.

    Parameters:
    vector (array-like): The input data array.
    value (float, optional): The central value around which to remove elements. Default is 0.
    window (float, optional): The range around the central value within which elements are considered for removal. Default is 0.4.

    Returns:
    array-like: Indices of the elements to be deleted.
    """
    # Calculate the upper and lower thresholds around the specified value
    top_threshold = value + window
    lower_threshold = value - window    
    # Find the indices of elements within the specified range
    index = np.where((vector > lower_threshold) & (vector < top_threshold))[0]
    # Randomly shuffle the indices
    np.random.shuffle(index)
    # Select half of the indices for deletion
    index_to_delete = index[:int(0.5 * index.shape[0])]
    # Return the indices to be deleted
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

def momentos(vector, order = 4):
  """
    Calculate the moments of a vector using different weight functions.

    Parameters:
    vector (array-like): The input data array with shape (Nev, Nt, Nc),
                         where Nev is the number of events, Nt is the number of time points,
                         and Nc is the number of channels.

    Returns:
    array-like: An array of moments calculated using different weight functions. The shape of the 
                returned array is (Nev, number_of_moments, Nc).
  """

  Nev,Nt,Nc = np.shape(vector)    #Nev: Núm eventos, Nt: Núm puntos temporales, Nc: Número canales
  t = np.reshape(np.linspace(0,Nt, Nt)/float(Nt),(1,-1,1)) #Normalized array of time
  MOMENT = np.zeros((Nev,0,Nc))

  for i in range(order): #Number of moments used
    W = t**(i) 
    W = np.tile(W,(Nev,1,Nc))
    MM = np.sum(vector*W,axis=1,keepdims=True)
    MOMENT = np.append(MOMENT,MM,axis=1)

    #W = np.exp(-(i)*t)
    #W = np.tile(W,(Nev,1,Nc))
    #MM = np.sum(vector*W,axis=1,keepdims=True)
    #MOMENT = np.append(MOMENT,MM,axis=1)

    #W = np.exp(-(t**(i)))
    #W = np.tile(W,(Nev,1,Nc))
    #MM = np.sum(vector*W,axis=1,keepdims=True)
    #MOMENT = np.append(MOMENT,MM,axis=1)

  return MOMENT

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def normalize_by_max(array_pulsos, fit_polynomial=True):
    """
    Normalize pulse data by the maximum value, optionally fitting a polynomial 
    for better normalization.

    Parameters:
    array_pulsos (numpy.ndarray): 3D array of pulse data with shape (n_pulses, n_samples, n_channels).
    fit_polynomial (bool): If True, fit a polynomial to a window around the maximum 
                           value before normalization. Defaults to True.

    Returns:
    numpy.ndarray: Normalized pulse data with the same shape as input.
    """
    # Initialize the output array with zeros, same shape as input
    y = np.zeros_like(array_pulsos)
    
    if fit_polynomial:
        # Loop over each pulse
        for i in range(array_pulsos.shape[0]):
            # Find the index of the maximum value in each channel
            index_max_channel0 = np.argmax(array_pulsos[i, :, 0])
            index_max_channel1 = np.argmax(array_pulsos[i, :, 1])
        
            # Define the window around the maximum value
            lower_window_channel0 = max(index_max_channel0 - 30, 0)
            lower_window_channel1 = max(index_max_channel1 - 30, 0)
            higher_window_channel0 = min(index_max_channel0 + 30, array_pulsos.shape[1])
            higher_window_channel1 = min(index_max_channel1 + 30, array_pulsos.shape[1])

            # Extract the values within the window for each channel
            y_channel0 = array_pulsos[i, lower_window_channel0:higher_window_channel0, 0]
            y_channel1 = array_pulsos[i, lower_window_channel1:higher_window_channel1, 1]
        
            # Create the x values corresponding to the window
            x_channel0 = np.arange(lower_window_channel0, higher_window_channel0)
            x_channel1 = np.arange(lower_window_channel1, higher_window_channel1)

            # Fit a 2nd-degree polynomial to the data in the window
            r_channel0 = np.polyfit(x_channel0, y_channel0, 2)
            r_channel1 = np.polyfit(x_channel1, y_channel1, 2)
        
            # Calculate the polynomial values
            y_channel0 = r_channel0[0]*x_channel0**2 + r_channel0[1]*x_channel0 + r_channel0[2]
            y_channel1 = r_channel1[0]*x_channel1**2 + r_channel1[1]*x_channel1 + r_channel1[2]
        
            # Normalize the original pulse data by the maximum value of the fitted polynomial
            y[i, :, 0] = array_pulsos[i, :, 0] / np.max(y_channel0)
            y[i, :, 1] = array_pulsos[i, :, 1] / np.max(y_channel1)
    
    else:
        # If no polynomial fitting is required, normalize directly by the maximum value in each channel
        for i in range(array_pulsos.shape[0]):
            y[i, :, 0] = array_pulsos[i, :, 0] / np.max(array_pulsos[i, :, 0])
            y[i, :, 1] = array_pulsos[i, :, 1] / np.max(array_pulsos[i, :, 1])
    
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


def normalize(data, method = 'standardization'):
    """
    Normalizes the data using the specified method and returns the normalized data along with the parameters.
    
    Parameters:
        data (numpy.ndarray): The data to be normalized (shape: N x M x 2).
        method (str): The normalization method ('min-max', 'max', 'standardization'). Default is 'standardization'.
        
    Returns:
        tuple: The normalized data, normalization parameters (depends on the method).
    """
    if method not in ['min-max', 'max', 'standardization']:
        raise ValueError("Invalid method. Choose from 'min-max', 'max', 'standardization'.")

    if method == 'min-max':
        min_vals = np.min(data[:, :, 0], axis=0)
        max_vals = np.max(data[:, :, 0], axis=0)
        normalized_data_dec0 = (data[:, :, 0] - min_vals) / (max_vals - min_vals)
        normalized_data_dec1 = (data[:, :, 1] - min_vals) / (max_vals - min_vals)
        params = (min_vals, max_vals)   

    elif method == 'max':
        max_vals = np.max(data[:, :, 0], axis=0)
        params = max_vals
        normalized_data_dec0 = data[:, :, 0] / max_vals
        normalized_data_dec1 = data[:, :, 1] / max_vals
    
    elif method == 'standardization':
        means = np.mean(data[:, :, 0], axis=0)
        stds = np.std(data[:, :, 0], axis=0)
        params = (means, stds)
        normalized_data_dec0 = (data[:, :, 0] - means) / stds
        normalized_data_dec1 = (data[:, :, 1] - means) / stds
        params = (means, stds)

    # Concatenate the normalized channels back together
    normalized_data = np.stack((normalized_data_dec0, normalized_data_dec1), axis=-1)
    
    return normalized_data, params


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def normalize_given_params(data, params, channel=0, method='standardization'):
    """
    Normalize the given data using the specified method and parameters.

    Parameters:
    data (array-like): The input data array with shape (N, M, Nc), where N is the number of events,
                       M is the number of time points, and Nc is the number of channels.
    params (tuple): A tuple containing the parameters needed for normalization. For 'min-max' method,
                    it should be (min_values, max_values). For 'standardization' method, it should be 
                    (mean_values, std_devs). Both min_values and max_values, or mean_values and std_devs should have 
                    lengths equal to M (second dimension of data).
    channel (int): The channel to normalize. Defaults to 0.
    method (str): The normalization method to use. Choose from 'min-max' or 'standardization'. Defaults to 'standardization'.

    Returns:
    array-like: The normalized data array with the same shape as the input data.
    
    Raises:
    ValueError: If the method is not one of 'min-max' or 'standardization'.
    ValueError: If params is not a tuple with two elements.
    ValueError: If the lengths of params[0] and params[1] do not match the second dimension (M) of data.
    """
    
    
    if method not in ['min-max', 'standardization']:
        raise ValueError("Invalid method. Choose from 'min-max' or 'standardization'.")
    
    # Check if params is a tuple and has two elements
    if not isinstance(params, tuple) or len(params) != 2:
        raise ValueError("Params must be a tuple with two elements.")
    
    if len(params[0]) != data.shape[1] or len(params[1]) != data.shape[1]:
        raise ValueError("Length of params[0] and params[1] must match the second dimension (axis = 1) of data.")
    
    # Create a copy of the original data to avoid modifying it
    data_copy = np.copy(data)
    
    if method == 'min-max':
        normalized_data = (data_copy[:, :, channel] - params[0]) / (params[1] - params[0])
    elif method == 'standardization':
        normalized_data = (data_copy[:, :, channel] - params[0]) / params[1]

    return normalized_data

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

from scipy import signal

def get_correlation(ref_pulse, pulse_set, channel=0):
    """
    Calculate the correlation of a reference pulse with every pulse in a set.

    Parameters:
    ref_pulse (array-like): The reference pulse to compare against.
    pulse_set (array-like): A set of pulses to search through. Expected shape is (num_pulses, pulse_length, num_channels).
    channel (int, optional): The channel of the pulses to use for comparison. Default is 0.

    Returns:
    array-like: An array of correlation values between the reference pulse and each pulse in the set.
    """

    y1 = ref_pulse
    n = len(y1)
    correlation = []

    for i in range(pulse_set.shape[0]):
        
        y2 = pulse_set[i, :, channel]
        corr = signal.correlate(y2, y1, mode = 'same')
        correlation.append(corr[n // 2])  # Append the correlation at delay zero to the list
    correlation = np.array(correlation)
    return correlation


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def get_closest(ref_pulse, pulse_set, channel = 0):
  """
    Calculate the index of the pulse in a set that is most similar to a reference pulse.

    Parameters:
    ref_pulse (array-like): The reference pulse to compare against.
    pulse_set (array-like): A set of pulses to search through. Expected shape is (num_pulses, pulse_length, num_channels).
    channel (int, optional): The channel of the pulses to use for comparison. Default is 0.

    Returns:
    int: The index of the pulse in pulse_set that is most similar to ref_pulse.
    """
  
  y1 = ref_pulse
  mse = []

  for i in range(pulse_set.shape[0]):
    y2 = pulse_set[i,:,channel]
    mse.append(np.mean((y1-y2)**2))
  
  mse = np.array(mse)
  sorted_indices = np.argsort(mse)
  index_of_closest = sorted_indices[1]  # Get the index of the closest pulse, excluding the first one (which is the reference pulse itself)

  return index_of_closest

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_set(og_set, channel = 0):
    """
    Create a new set of pulses where each pulse is paired with its closest match from the original set.

    Parameters:
    og_set (array-like): The original set of pulses. Expected shape is (num_pulses, pulse_length, num_channels).
    channel (int, optional): The channel of the pulses to use for finding the closest match. Default is 0.

    Returns:
    array-like: A new set of pulses where each pulse in the original set is paired with its closest match.
                The returned set has shape (num_pulses, pulse_length, 2), where the first channel is the original pulse
                and the second channel is the closest matching pulse.
    """
    
    new_set = np.zeros_like(og_set)
  
    for i in range(og_set.shape[0]):
        
        closest = get_closest(og_set[i, :, channel], og_set, channel = channel)  # Find the index of the closest pulse to the current pulse
        new_set[i, :, 0] = og_set[i, :, channel]         # Assign the original pulse to the first channel of the new set
        new_set[i, :, 1] = og_set[closest, :, channel] # Assign the closest matching pulse to the second channel of the new set

    return new_set

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_position(pulse_set, channel_to_move = 1, channel_to_fix = 0, t_shift = 8, NOISE = True):
    """
    Create a new position for the pulse set by shifting one channel and optionally adding noise.

    Parameters:
    pulse_set (np.ndarray): The input pulse set array of shape (N_pulse_pairs, n_time_points, n_channels).
    channel_to_move (int): The index of the channel to be shifted. Default is 1.
    channel_to_fix (int): The index of the channel to remain fixed. Default is 0.
    t_shift (int): The number of time points to shift the channel. Default is 8.
    NOISE (bool): Whether to add noise to the shifted channel. Default is True.

    Returns:
    np.ndarray: The new pulse set array with the specified channel shifted and optionally noise added.
    """

    New_position = np.zeros_like(pulse_set)
    
    for i in range(New_position.shape[0]):
        
        New_position[i,:,channel_to_fix] = pulse_set[i,:,channel_to_fix]
        New_position[i,:,channel_to_move] = np.roll(pulse_set[i,:,channel_to_move], t_shift)
        
        if NOISE:
            noise00 = np.random.normal(scale = 1e-3, size = t_shift)
            noise0 = np.random.normal(scale = 0.01, size = New_position.shape[1])
            smoothed_noise = gaussian_filter1d(noise0, sigma = 10)
            New_position[i,:,channel_to_move] = New_position[i,:,channel_to_move]  + smoothed_noise
            New_position[i,:t_shift,channel_to_move] = noise00
        else:
            New_position[i,:t_shift,channel_to_move] = pulse_set[i,:t_shift,channel_to_move]
    
    return New_position


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


def pulso_sigmoid(t,t0, A=100, center_window=0.3, rise_window=0.2, tau_rise = 15, tau_drop = 150, NOISE = True):

  if NOISE:
    t0 = t0 + np.random.uniform(-center_window, center_window)
    tau_rise = tau_rise + np.random.uniform(-rise_window, rise_window)
    noise = np.random.normal(scale = 0.01, size = len(t))
    smoothed_noise = gaussian_filter1d(noise, sigma = 10)
    y = A*(1/(1 + np.exp(-(t-t0)/tau_rise)))*np.exp(-(t-t0)/tau_drop) 
    y = y + smoothed_noise
  
  else:
    y = A*(1/(1 + np.exp(-(t-t0)/tau_rise)))*np.exp(-(t-t0)/tau_drop)  
  
  y = y / np.max(y)
  return y


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def pulso_poisson(t, t0, A=1000, center_window=0.3, rise_window=0.2, tau_rise=15, tau_drop=150, NOISE=True):
    """
    Generate a pulse with optional noise.

    Parameters:
    - t (array-like): The time array.
    - t0 (float): The central time around which the pulse is centered.
    - A (float, optional): The amplitude of the pulse. Default is 1000.
    - center_window (float, optional): The range around t0 to randomly vary the central time. Default is 0.3.
    - rise_window (float, optional): The range to randomly vary the rise time. Default is 0.2.
    - tau_rise (float, optional): The rise time constant. Default is 15.
    - tau_drop (float, optional): The drop time constant. Default is 150.
    - NOISE (bool, optional): Whether to add noise to the pulse. Default is True.

    Returns:
    array-like: The generated pulse.
    """
   
    centro = t0 + np.random.uniform(-center_window, center_window)
    tau_rise = tau_rise + np.random.uniform(-rise_window, rise_window)
    y = A * (1 - np.exp(-(t - centro) / tau_rise)) * np.exp(-(t - centro) / tau_drop)
    y[y < 0.] = 0.
    
    if NOISE:
        noise = np.random.normal(scale=0.01, size=len(t))
        smoothed_noise = gaussian_filter1d(noise, sigma=100)
        y = y + np.random.poisson(y) + smoothed_noise
    
    y = y / np.max(y)
    return y

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def pulso_escalon(t,t0, A = 1):
  y = np.zeros_like(t) 
  y[:t0] = 0.
  y[t0:] = A
  return y

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def delay_pulse_pair(pulse_set, time_step, t_shift = 0, delay_steps=32, NOISE=True):
    """
    Function to apply random delays to each pulse in a pair, calculate a reference time difference,
    and optionally add noise to the delayed pulses.

    Parameters:
    - pulse_set (array): Array of pulse pairs. Shape is (N, M, 2) where N is number of pulse pairs,
                         M is number of time points, and '2' indicates two channels.
    - time_step (float): The time interval between consecutive time points in the data.
    - t_shift (int, optional): Initial time shift applied between two channels. Positive values indicate
                               that channel 1 is delayed relative to channel 0, and negative values the opposite.
    - delay_steps (int, optional): Maximum number of delay steps to apply.
    - NOISE (bool, optional): If True, random Gaussian noise is added to the beginning of the pulses.

    Returns:
    - INPUT (array): Array containing the delayed pulses with the same shape as pulse_set.
    - REF (array): Array of reference time differences adjusted for the initial shift, t_shift.

    """
    
    INPUT = np.zeros_like(pulse_set)
    REF = np.zeros((pulse_set.shape[0],), dtype=np.float32)

    NRD0 = np.random.randint(delay_steps, size=pulse_set.shape[0])
    NRD1 = np.random.randint(delay_steps, size=pulse_set.shape[0])

    for i in range(pulse_set.shape[0]):
        N0 = NRD0[i]
        INPUT[i, :, 0] = np.roll(pulse_set[i, :, 0], N0)

        N1 = NRD1[i]
        INPUT[i, :, 1] = np.roll(pulse_set[i, :, 1], N1)

        # Calculate the reference time difference taking into account the delays and initial time shift.
        REF[i] = time_step * (N0 - N1 - t_shift)

        # Add noise to the beginning of the pulses, if enabled.
        if NOISE:
            noise0 = np.random.normal(scale=1e-3, size=N0)
            noise1 = np.random.normal(scale=1e-3, size=N1)
            INPUT[i, 0:N0, 0] = noise0
            INPUT[i, 0:N1, 1] = noise1
        else:
            # If no noise, retain the original pulses in the delayed sections.
            INPUT[i, 0:N0, 0] = pulse_set[i,0:N0, 0]
            INPUT[i, 0:N1, 1] = pulse_set[i,0:N1, 1]

    return INPUT, REF

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_and_delay_pulse_pair(pulse_set, time_step, delay_steps = 32, NOISE = True):
    """
    Function to create pulses pairs and apply random delays to each other, calculate a reference 
    time difference, and optionally add noise to the delayed pulses.

    Parameters:
    - pulse_set (array): Array of pulses. Shape is (N, M) where N is number of pulses,
                         M is number of time points.
    - time_step (float): The time interval between consecutive time points in the data.
    - delay_steps (int, optional): Maximum number of delay steps to apply.
    - NOISE (bool, optional): If True, random Gaussian noise is added to the beginning of the pulses.

    Returns:
    - INPUT (array): Array containing the delayed pulses with the same shape as pulse_set.
    - REF (array): Array of reference time differences adjusted for the initial shift, t_shift.

    """
    
    INPUT = np.zeros((pulse_set.shape[0],pulse_set.shape[1],2))
    REF = np.zeros((pulse_set.shape[0],), dtype = np.float32)

    NRD0 = np.random.randint(delay_steps, size = pulse_set.shape[0])
    NRD1 = np.random.randint(delay_steps, size = pulse_set.shape[0])

    for i in range(pulse_set.shape[0]):
        N0 = NRD0[i]
        INPUT[i, :, 0] = np.roll(pulse_set[i, :], N0)

        N1 = NRD1[i]
        INPUT[i, :, 1] = np.roll(pulse_set[i, :], N1)

        # Calculate the reference time difference taking into account the delays and initial time shift.
        REF[i] = time_step * (N0 - N1)

        # Add noise to the beginning of the pulses, if enabled.
        if NOISE:
            noise0 = np.random.normal(scale=1e-3, size=N0)
            noise1 = np.random.normal(scale=1e-3, size=N1)
            INPUT[i, 0:N0, 0] = noise0
            INPUT[i, 0:N1, 1] = noise1
        else:
            # If no noise, retain the original pulses in the delayed sections.
            INPUT[i, 0:N0, 0] = pulse_set[i,0:N0]
            INPUT[i, 0:N1, 1] = pulse_set[i,0:N1]

    return INPUT, REF


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def delay_pulse_4_channels(pulse_set, time_step, delay_steps = 20, NOISE = True):
  """
    Delays pulses and optionally adds noise to create reference time differences between channels.
    
    Parameters:
        pulse_set (np.array): Array of pulse signals.
        time_step (float): Time step value for calculating reference time differences.
        delay_steps (int): Maximum delay steps for rolling the pulses.
        NOISE (bool): Flag to add Gaussian noise to the signals.
    
    Returns:
        np.array: Modified pulse signals with delays and optional noise.
        np.array: Time differences between channel 0 and 1.
        np.array: Time differences between channel 2 and 3.
  """

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
    """
    Calculate the mean pulse from a set of pulses using Fourier transforms.

    Parameters:
    pulse_set (array-like): The input set of pulses. Expected shape is (num_pulses, pulse_length, num_channels).
    channel (int, optional): The channel of the pulses to use for calculation. Default is 0.

    Returns:
    array-like: The mean pulse calculated from the set of pulses.
    """
    transforms = []
    
    for i in range(pulse_set.shape[0]):
        fourier_transform = np.fft.fft(pulse_set[i, :, channel])
        transforms.append(fourier_transform)
    
    transforms = np.array(transforms, dtype='object')
    sum_of_transf = np.sum(transforms, axis = 0)
    reconstructed_signal = np.fft.ifft(sum_of_transf)
    normalized_reconstructed_signal = reconstructed_signal / np.max(reconstructed_signal)
    mean_pulse = np.real(normalized_reconstructed_signal)
  
    return mean_pulse

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def move_to_reference(reference, pulse_set, start=50, stop=80, max_delay=10, channel=0):
    """
    Aligns each pulse in a set with a reference pulse by shifting it to minimize MSE.
    
    Parameters:
        reference (np.array): The reference pulse.
        pulse_set (np.array): The set of pulses to align.
        start (int): Start index for slicing the pulses.
        stop (int): Stop index for slicing the pulses.
        max_delay (int): Maximum delay allowed for shifting.
        channel (int): Channel index to use from pulse_set.
    
    Returns:
        np.array: Array of delay steps for each pulse to achieve minimum MSE.
        np.array: Array of moved pulses corresponding to the minimal MSE alignment.
    """

    if int(stop-start) < max_delay:
       print('Window (stop-start) cannot be smaller than max_delay')

    y1 = reference[start:stop]
    delays = []
    moved_pulses = []
    for i in range(pulse_set.shape[0]):
        mse = []
        y2_list = []
        y2 = pulse_set[i, start:stop, channel]
        for j in range(-max_delay, max_delay + 1):  # j goes from -max_delay to max_delay
            y2_rolled = np.roll(y2, j)
            # Correct edges based on shift direction
            if j < 0:
                y2_rolled[j:] = pulse_set[i, stop:stop + abs(j), channel]
            if j >= 0:
                y2_rolled[:j] = pulse_set[i, :j, channel]
            mse.append(np.mean((y1 - y2_rolled)**2))
            y2_list.append(y2_rolled)
        
        mse = np.array(mse)
        min_mse_index = np.argmin(mse)
        delay_steps = min_mse_index - max_delay  # adjust index to reflect actual shift
        delays.append(delay_steps)
        
        y2_array = np.array(y2_list)
        moved_pulses.append(y2_array[min_mse_index])  # Reuse min_mse_index to avoid recomputation

    return np.array(delays), np.array(moved_pulses)

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
    
def cut_pulse_by_fraction(vector, fraction=0.2, window_low=140, window_high=10):
    """
    Truncates pulse data in the input vector based on a specified fraction.

    Parameters:
    vector (ndarray): Input 3D array of shape (n_samples, n_points, n_channels).
    fraction (float): Fraction threshold to determine the start of the pulse. Default is 0.2.
    window_low (int): Number of points before the fraction threshold to retain. Default is 140.
    window_high (int): Number of points after the fraction threshold to retain. Default is 10.

    Returns:
    ndarray: A new vector with truncated pulse data.
    """
    new_vector = np.copy(vector)
    
    for i in range(vector.shape[0]):
        # Find indices where the signal in each channel exceeds the fraction threshold
        indices_channel0 = np.where(vector[i,:, 0] >= fraction)[0]
        indices_channel1 = np.where(vector[i,:, 1] >= fraction)[0]
        
        # Calculate the low and high indices to truncate around the fraction threshold
        low_index_channel0 = indices_channel0[0] - window_low
        low_index_channel1 = indices_channel1[0] - window_low

        high_index_channel0 = indices_channel0[0] + window_high
        high_index_channel1 = indices_channel1[0] + window_high
        
        # Set values outside the specified windows to zero for each channel
        new_vector[i,:low_index_channel0, 0] = 0.0
        new_vector[i,:low_index_channel1, 1] = 0.0
        
        new_vector[i,high_index_channel0:, 0] = 0.0
        new_vector[i,high_index_channel1:, 1] = 0.0
    
    return new_vector    

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def calculate_gaussian_center_sigma(vector, shift, nbins = 51):
    """
    Calculate the Gaussian fit parameters (centroid and standard deviation) for each row of the input vector.

    Parameters:
    vector (numpy.ndarray): Input 2D array where each row represents a set of data points.
    shift (numpy.ndarray): Array of shift values to be subtracted from each row of the input vector.
    nbins (int, optional): Number of bins to use for the histogram. Default is 51.

    Returns:
    numpy.ndarray: Array of centroid values for each row of the input vector.
    numpy.ndarray: Array of standard deviation values for each row of the input vector.
    """
    
    # Initialize lists to store the centroid and standard deviation for each row
    centroid = []
    std = []
    
    # Loop over each row in the input vector
    for i in range(vector.shape[0]):
        # Calculate the histogram of the current row after applying the shift
        histogN, binsN = np.histogram(vector[i, :] - shift[i], bins=nbins, range=[-0.8, 0.8])
        
        # Calculate the center of each bin
        cbinsN = 0.5 * (binsN[1:] + binsN[:-1])
        
        try:
            # Perform Gaussian fitting
            HN, AN, x0N, sigmaN = gauss_fit(cbinsN, histogN)
            
            # Handle cases where sigmaN is NaN
            if np.isnan(sigmaN):
                sigmaN = 10
                x0N = 10
        except:
            # Handle exceptions by setting default values
            x0N, sigmaN = 10, 10
        
        # Append the results to the respective lists
        centroid.append(x0N)
        std.append(sigmaN)
    
    # Convert lists to numpy arrays
    centroid = np.array(centroid, dtype='float64')
    std = np.array(std, dtype='float64')
    
    # Return the results
    return centroid, std