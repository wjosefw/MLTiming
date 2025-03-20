import torch
import random
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

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

def calculate_gaussian_center(vector, nbins = 51, limit = 1):
    """
    Calculate Gaussian fit centroid for each row of the input vector.

    Parameters:
    vector (numpy.ndarray): A 2D array where each row represents a set of data points to be analyzed.
    nbins (int, optional): Number of bins to use for the histogram.
    limits (float, optional): Range of the histogram. 

    Returns:
    numpy.ndarray: Array of centroid values (x0) for each row of the input vector. 
    """
    
    centroid = [] 

    for i in range(vector.shape[0]):
        # Calculate the histogram of the current row of data
        histogN, binsN = np.histogram(vector[i, :], bins = nbins, range = [-limit,limit]) 
        
        # Calculate the bin centers
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
            
        centroid.append(x0N)
    
    centroid = np.array(centroid, dtype='float64')

    return centroid

def plot_gaussian(array, shift, nbins = 51, label = ' ', range = 0.8, limit = 1):
    """
    Plot histogram centered around its peak and overlay Gaussian fit.
    """

    # Compute the initial histogram to find the peak
    histog, bins = np.histogram(array - shift, bins = nbins, range = [-limit,limit])
    cbins = 0.5 * (bins[1:] + bins[:-1])

    # Find peak position
    peak_index = np.argmax(histog)
    peak_value = cbins[peak_index]
    
    # Define a histogram range centered at the peak
    new_min, new_max = peak_value - range, peak_value + range

    # Compute the histogram with the new centered range
    histog, bins, patches = plt.hist(array - shift, bins=nbins, range=(new_min, new_max), alpha=0.5, label=label)
    cbins = 0.5 * (bins[1:] + bins[:-1])

    # Fit a Gaussian to the histogram data
    popt = gauss_fit(cbins, histog)
    hist_color = patches[0].get_facecolor()
    
    # Generate Gaussian fit curve
    x_fit = np.linspace(new_min, new_max, 500)
    y_fit = gauss(x_fit, *popt)
    
    plt.plot(x_fit, y_fit, color=hist_color)
    plt.legend()

def get_gaussian_params(array, shift, nbins = 51, range = 0.5, limit = 1):
    """
    Compute Gaussian parameters for a histogram centered around its peak.
    """
    
    # Compute an initial histogram to find the peak
    histog_full, bins_full = np.histogram(array - shift, bins = nbins, range =[-limit, limit])
    cbins_full = 0.5 * (bins_full[1:] + bins_full[:-1])

    # Identify peak bin
    peak_index = np.argmax(histog_full)
    peak_value = cbins_full[peak_index]

    # Define a histogram range centered around the peak
    new_min, new_max = peak_value - range, peak_value + range

    # Compute the histogram within the new range
    histog, bins = np.histogram(array - shift, bins = nbins, range = (new_min, new_max))
    cbins = 0.5 * (bins[1:] + bins[:-1])  # Bin centers

    # Fit a Gaussian to the histogram data
    x, y = cbins, histog
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    try:
        popt, pcov = curve_fit(gauss, x, y, p0 = [min(y), max(y), mean, sigma])
        HN, AN, x0, sigma = popt
        FWHM = 2.35482 * sigma
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan] * 3)
    except:
        print("Gaussian fitting failed.")
        return None, None

    return (HN, x0, FWHM), perr


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

from numba import cuda

@cuda.jit
def calculate_moments_gpu(vector, t, MOMENT, order):
    Nev, Nt, Nc = vector.shape[:3]  # Get dimensions
    
    # Thread indices in the CUDA grid
    event_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    order_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if event_idx < Nev and order_idx < order:
        for channel in range(Nc):
            moment = 0.0
            for time_idx in range(Nt):
                moment += vector[event_idx, time_idx, channel] * (t[time_idx] ** order_idx)
            MOMENT[event_idx, order_idx, channel] = moment

def momentos(vector, order = 4):
    """
    Calculate the moments of a vector using CUDA with Numba for GPU acceleration.

    Parameters:
    vector (array-like): The input data array with shape (Nev, Nt, Nc).
                         where Nev is the number of events, Nt is the number of time points,
                         and Nc is the number of channels.

    Returns:
    array-like: An array of moments calculated using different weight functions.
                The shape of the returned array is (Nev, order, Nc).
    """
    # Ensure the array is contiguous
    vector = np.ascontiguousarray(vector)

    Nev, Nt, Nc = vector.shape
    t = np.linspace(0, 1, Nt)  # Normalized time array

    # Allocate memory on the GPU
    vector_gpu = cuda.to_device(vector)
    t_gpu = cuda.to_device(t)
    MOMENT_gpu = cuda.device_array((Nev, order, Nc))  # Create an empty array on the GPU to store results

    # Define the CUDA grid and block size
    threads_per_block = (16, 16)  # Number of threads per block (adjust as needed)
    blocks_per_grid = ((Nev + threads_per_block[0] - 1) // threads_per_block[0],
                       (order + threads_per_block[1] - 1) // threads_per_block[1])

    # Launch the kernel to calculate moments on the GPU
    calculate_moments_gpu[blocks_per_grid, threads_per_block](vector_gpu, t_gpu, MOMENT_gpu, order)

    # Copy the result back to the CPU
    MOMENT = MOMENT_gpu.copy_to_host()

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
    n = y.shape[1]

    # Handle the even and odd index multiplications
    even_indices = np.arange(2, n - 1, 2)  # Indices for 2y_i terms (even indices)
    odd_indices = np.arange(1, n, 2)       # Indices for 4y_i terms (odd indices)

    # Apply Simpson's rule across all rows at once
    # Add y[0] + y[-1]
    integral = y[:, 0] + y[:, -1]

    # Add 4 * y[odd_indices]
    integral += 4 * np.sum(y[:, odd_indices], axis=1)

    # Add 2 * y[even_indices]
    integral += 2 * np.sum(y[:, even_indices], axis=1)

    # Multiply by h/3 and return the result
    return integral * h / 3

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def normalize(data, method = 'standardization'):

    if method not in ['min-max', 'max', 'standardization']:
        raise ValueError("Invalid method. Choose from 'min-max', 'max', 'standardization'.")

    if method == 'min-max':
        if len(data.shape) == 3:
            min_vals = np.min(data[:, :, 0], axis=0)
            max_vals = np.max(data[:, :, 0], axis=0)
            normalized_data_dec0 = (data[:, :, 0] - min_vals) / (max_vals - min_vals)
            normalized_data_dec1 = (data[:, :, 1] - min_vals) / (max_vals - min_vals)
            normalized_data = np.stack((normalized_data_dec0, normalized_data_dec1), axis=-1) # Concatenate the normalized channels back togethe
            params = (min_vals, max_vals)   
        
        if len(data.shape) == 2:
            max_vals = np.max(data, axis = 0)
            min_vals = np.min(data, axis = 0)
            normalized_data = (data - min_vals) / (max_vals - min_vals)
            params = (min_vals, max_vals)   

    elif method == 'max':
        if len(data.shape) == 3:
            max_vals = np.max(data[:, :, 0], axis=0)
            params = max_vals
            normalized_data_dec0 = data[:, :, 0] / max_vals
            normalized_data_dec1 = data[:, :, 1] / max_vals
            normalized_data = np.stack((normalized_data_dec0, normalized_data_dec1), axis=-1)
        
        if len(data.shape) == 2:
            max_vals = np.max(data, axis=0)
            params = max_vals
            normalized_data = data / max_vals
    
    elif method == 'standardization':
        if len(data.shape) == 3:
            means = np.mean(data[:, :, 0], axis=0)
            stds = np.std(data[:, :, 0], axis=0)
            normalized_data_dec0 = (data[:, :, 0] - means) / stds
            normalized_data_dec1 = (data[:, :, 1] - means) / stds
            normalized_data = np.stack((normalized_data_dec0, normalized_data_dec1), axis=-1)
            params = (means, stds)
        
        if len(data.shape) == 2:
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            normalized_data = (data - means) / stds
            params = (means, stds)

    
    return normalized_data, params

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

def create_positive_and_negative_delays(pulse_set, time_step, start = 50, stop = 74, delay_time = 1):
    

    INPUT = np.zeros((pulse_set.shape[0], int(stop-start), 2))
    INPUT_2 = np.zeros((pulse_set.shape[0], int(stop-start), 2))
    REF = np.zeros((pulse_set.shape[0],), dtype = np.float32)

    NRD0 = np.random.uniform(low = -0.05, high = delay_time, size = pulse_set.shape[0])
    NRD1 = np.random.uniform(low = -0.05, high = delay_time, size = pulse_set.shape[0])
    
    for i in range(pulse_set.shape[0]):
        
        if NRD0[i] >= 0:
            res_0 = NRD0[i] % time_step  # Fractional part of the delay
            for j in range(int(stop-start) - 1, 0, -1):
                slope = (pulse_set[i, start + j] - pulse_set[i, start + j - 1]) / time_step
                INPUT[i, j, 0] = pulse_set[i, start + j] - slope * res_0 
            INPUT[i, 0, 0] = pulse_set[i, start] 

            idel_0 = int( NRD0[i] / time_step) 
            INPUT_2[i,:,0] = np.roll(INPUT[i,:,0], idel_0)
            INPUT_2[i,:idel_0,0] = INPUT[i,:idel_0,0]
        
        if NRD0[i] < 0:
            res_0 = NRD0[i] % time_step  # Fractional part of the delay
            for j in range(int(stop-start) - 1):
                slope = (pulse_set[i, start + j + 1] - pulse_set[i, start + j]) / time_step
                INPUT[i, j, 0] = pulse_set[i, start + j] + slope * res_0 
            INPUT[i, -1, 0] = pulse_set[i, stop] 

            idel_0 = int(NRD0[i] / time_step) 
            if idel_0 <= -1:
                INPUT_2[i,:,0] = np.roll(INPUT[i,:,0], idel_0)
                INPUT_2[i,idel_0:,0] = pulse_set[i, stop + 1 : stop + abs(idel_0) + 1]
            else:
                INPUT_2[i,:,0] = INPUT[i,:,0]

        if NRD1[i] >= 0:
            res_1 = NRD1[i] % time_step  
            for j in range(int(stop-start) - 1, 0, -1):
                slope = (pulse_set[i, start + j] - pulse_set[i, start + j - 1]) / time_step
                INPUT[i, j, 1] = pulse_set[i, start + j] - slope * res_1 
            INPUT[i, 0, 1] = pulse_set[i, start] 

            idel_1 = int( NRD1[i] / time_step) 
            INPUT_2[i,:,1] = np.roll(INPUT[i,:,1], idel_1)
            INPUT_2[i,:idel_1,1] = INPUT[i,:idel_1,1]
        
        if NRD1[i] < 0:
           res_1 = NRD1[i] % time_step  # Fractional part of the delay
           for j in range(int(stop-start) - 1):
               slope = (pulse_set[i, start + j + 1] - pulse_set[i, start + j]) / time_step
               INPUT[i, j, 1] = pulse_set[i, start + j] + slope * res_1 
           INPUT[i, -1, 1] = pulse_set[i, stop] 
           
           idel_1 = int(NRD1[i] / time_step) 
           if idel_1 != 0:
            INPUT_2[i,:,1] = np.roll(INPUT[i,:,1], idel_1)
            INPUT_2[i,idel_1:,1] = pulse_set[i, stop + 1 : stop + abs(idel_1) + 1]
           else:
            INPUT_2[i,:,1] = INPUT[i,:,1]
        
        REF[i] = NRD0[i] - NRD1[i]

    return INPUT_2, REF

def create_and_delay_pulse_pair(pulse_set, time_step, delay_time = 1):
    

    INPUT = np.zeros((pulse_set.shape[0], pulse_set.shape[1], 2))
    INPUT_2 = np.zeros((pulse_set.shape[0], pulse_set.shape[1], 2))
    REF = np.zeros((pulse_set.shape[0],), dtype = np.float32)

    NRD0 = np.random.uniform(low = 0, high = delay_time, size = pulse_set.shape[0])
    NRD1 = np.random.uniform(low = 0, high = delay_time, size = pulse_set.shape[0])
    
    for i in range(pulse_set.shape[0]):
        
        res_0 = NRD0[i] % time_step  # Fractional part of the delay
        for j in range(pulse_set.shape[1] - 1, 0, -1):
            slope = (pulse_set[i, j] - pulse_set[i, j-1]) / time_step
            INPUT[i, j, 0] = pulse_set[i, j] - slope * res_0 
        INPUT[i, 0, 0] = pulse_set[i, 0] 

        idel_0 = int( NRD0[i] / time_step) 
        INPUT_2[i,:,0] = np.roll(INPUT[i,:,0], idel_0)
        INPUT_2[i,:idel_0,0] = INPUT[i,:idel_0,0]


        res_1 = NRD1[i] % time_step  #
        for j in range(pulse_set.shape[1] - 1, 0, -1):
            slope = (pulse_set[i, j] - pulse_set[i, j-1]) / time_step
            INPUT[i, j, 1] = pulse_set[i, j] - slope * res_1 
        INPUT[i, 0, 1] = pulse_set[i, 1] 

        idel_1 = int( NRD1[i] / time_step) 
        INPUT_2[i,:,1] = np.roll(INPUT[i,:,1], idel_1)
        INPUT_2[i,:idel_1,1] = INPUT[i,:idel_1,1]
        
        REF[i] = NRD0[i] - NRD1[i]  

    return INPUT_2, REF

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

import cupy as cp
def move_to_reference(reference, pulse_set, start = 50, stop = 80, channel = 0):
    batch_size = 50000 # For an NVIDIA RTX 4090 reduce if smaller GPU
    
    window_size = stop - start
    reference_pulse = reference[start:stop]  # Extract reference pulse
    num_pulses = pulse_set.shape[0]

    # Initialize results
    delays = cp.zeros(num_pulses, dtype = cp.int32)
    aligned_pulses = cp.zeros((num_pulses, window_size), dtype = cp.float32)

    # Convert reference pulse and pulse set to GPU
    reference_pulse = cp.asarray(reference_pulse, dtype = cp.float32)
    pulse_set = cp.asarray(pulse_set, dtype = cp.float32)

    # Process in batches to avoid memory overload
    for i in range(0, num_pulses, batch_size):
        batch_end = min(i + batch_size, num_pulses)
        batch_pulses = pulse_set[i:batch_end, :, channel]  # Select batch

        # Create sliding windows for the batch
        segments = cp.lib.stride_tricks.sliding_window_view(batch_pulses, window_shape = window_size, axis = 1).astype(cp.float32)

        # Compute Mean Squared Error (MSE)
        mse = cp.mean((segments - reference_pulse) ** 2, axis=2)

        # Find optimal shifts
        min_mse_indices = cp.argmin(mse, axis = 1)
        optimal_shifts = start - min_mse_indices

        # Assign batch results
        delays[i:batch_end] = optimal_shifts
        aligned_pulses[i:batch_end, :] = segments[cp.arange(batch_end - i), min_mse_indices, :]

        # Free GPU memory
        del batch_pulses, segments, mse, min_mse_indices, optimal_shifts
        cp.get_default_memory_pool().free_all_blocks()

    return delays.get(), aligned_pulses.get()
   
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def set_seed(seed):
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

def interpolate_pulses(data, EXTRASAMPLING = 8, time_step = 0.2):
    """
    Interpolate real pulses using cubic interpolation.

    - data: numpy array of shape (N, M, 2), where N is the number of pulses,
      M is the number of time points, and 2 represents the real and imaginary components.    
    """
    Nt = np.shape(data)[1]
    Nt_new = Nt * EXTRASAMPLING
    new_time_step = time_step / EXTRASAMPLING

    t = np.linspace(0, Nt, Nt)
    t_new = np.linspace(0, Nt, Nt_new)

    interp_func_data = interp1d(t, data, kind = 'cubic', axis = 1)
    new_data = interp_func_data(t_new)

    return new_data, new_time_step

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def constant_fraction_discrimination(vector, time_step = 0.2, fraction = 0.9, shift = 30, plot = True):
    corrected_signal = np.zeros_like(vector)
    delayed_signal = np.zeros_like(vector)
    delayed_signal2 = np.zeros_like(vector)
    
    for i in range(vector.shape[0]):
        
        res = shift % time_step  # Fractional part of the delay
        for j in range(vector.shape[1] - 1, 0, -1):
            slope = (vector[i, j] - vector[i, j-1]) / time_step
            delayed_signal[i, j] = vector[i, j] - slope * res 
        delayed_signal[i, 0] = vector[i, 0] 

        idel = int(shift / time_step) 
        delayed_signal2[i,:] = np.roll(delayed_signal[i,:], idel)
        delayed_signal2[i,:idel] = delayed_signal[i,:idel]
        
        fraction_signal = fraction*(-vector[i,:])
        corrected_signal[i,:] = delayed_signal2[i,:] + fraction_signal
        if plot:
            plt.plot(corrected_signal[i, :])
    return corrected_signal


def find_first_zero_crossing_after_minimum(vector, time_step):

    # Create time array
    t = np.arange(vector.shape[0]) * time_step 
    
    # Find index of array min
    min_index = np.argmin(vector)

    # Look for the first zero-crossing after the minimum
    for i in range(min_index, len(vector) -  1):
        if vector[i] < 0 and vector[i + 1] > 0:
            m = (vector[i + 1] - vector[i]) / (t[i + 1] - t[i])
            b = vector[i] - m*t[i]
            crossing_time = -b / m
            break  # Break the loop after finding the crossing point
        if vector[i] == 0:
            crossing_time = t[i]
            break
        else:
            crossing_time = 0 # Placeholder if the other conditions do not get fullfilled
         
    if min_index == len(vector) - 1:
        crossing_time = t[-1]
    
    return crossing_time               

def Calculate_CFD(array, fraction = 0.7, shift = 80, time_step = 0.025):
    """
    Calculate the timestamps of signals using Constant Fraction Discrimination (CFD) method.
    
    Parameters:
    ----------
    - Array : A 2D array representing the input signals where each row corresponds to a separate signal.
    - Fraction (float): The fraction of the maximum signal amplitude used for discrimination.
    - Shift (int): The amount to shift the CFD signal for discrimination.     
    - Time_step (float): The time step used for calculating timestamps from the CFD signal.
    
    Returns:
    -------
    A 2D array where each row contains the calculated timestamps for the corresponding input signal.
    """
    cfd_signal = constant_fraction_discrimination(array, fraction = fraction, shift = shift, plot = False)
    
    timestamps_list = []
    for i in range(cfd_signal.shape[0]):
        timestamps = find_first_zero_crossing_after_minimum(cfd_signal[i,:], time_step)
        timestamps_list.append(timestamps)
    
    return np.array(timestamps_list)


def extract_signal_window_by_LED(vector, time_step = 0.2, threshold = 0.1, window_low = 10, window_high = 7):
    """
    Extracts a window of signal data around a calculated timestamp for each signal in the input vector array.
    The timestamp is determined based on the LED threshold. 
    Interpolates the signal to align with the calculated LED position.
    """

    # Initialize arrays to store truncated and interpolated signal windows
    new_vector = np.zeros((vector.shape[0], int(window_high + window_low)))
    new_vector2 = np.zeros((vector.shape[0], int(window_high + window_low)))

    # Calculate and store LED timestamps for each signal
    timestamps_LED = []

    for i in range(vector.shape[0]):
        timestamp = calculate_slope_y_intercept(vector[i, :], time_step, threshold = threshold)
        timestamps_LED.append(timestamp)

    # Extract the signal window and perform interpolation
    for i in range(vector.shape[0]):
        # Calculate the index corresponding to the detected LED timestamp
        idel = int(timestamps_LED[i] / time_step)
        
        # Define the indices for the low and high parts of the window
        low_index = idel - window_low
        high_index = idel + window_high
        
        # Extract the signal window
        new_vector[i, :] = vector[i, low_index:high_index]
        
        # Calculate the residue and interpolate the signal
        res = timestamps_LED[i] % time_step
        for j in range(new_vector.shape[1] - 1):
            slope = (new_vector[i, j + 1] - new_vector[i, j]) / time_step
            new_vector2[i, j] = new_vector[i, j] + slope * res
        last_slope = (vector[i, high_index + 1] - vector[i, high_index]) / time_step
        new_vector2[i, -1] = new_vector[i, -1] + last_slope * res

    return new_vector2


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def calculate_slope_y_intercept(vector, time_step, threshold=0.1):
    """
    Calculate the time at which a vector exceeds a given threshold by linear interpolation.

    Parameters:
    ----------
    - Vector (numpy.ndarray): A 1D array representing the signal or data to analyze.
    - Time_step (float): The time interval between consecutive samples in the vector.  
    - Threshold (float): The threshold value to identify the crossing point.

    Returns:
    -------
    time (float): The estimated time at which the signal first exceeds the threshold value. 
    """
    
    t = np.arange(vector.shape[0]) * time_step
    index = np.where(vector > threshold)[0][0]
    t1 = t[index]
    t0 = t[index - 1]
    m = (vector[index] - vector[index - 1]) / (t1 - t0)
    b = vector[index - 1] - m*t0
    time = (threshold - b) / m
    return time

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def continuous_delay(vector, time_step = 0.2, delay_time = 1, channel_to_fix = 0, channel_to_move = 0):
    
    res = delay_time % time_step  # Fractional part of the delay
    idel = int(delay_time / time_step) 
    
    new_vector = np.zeros_like(vector)
    if delay_time >= 0:
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1] - 1, 0, -1):
                    slope = (vector[i, j, channel_to_move] - vector[i, j - 1, channel_to_move]) / time_step
                    new_vector[i, j, channel_to_move] =  vector[i, j, channel_to_move] - slope * res 
            new_vector[i,0, channel_to_move] = vector[i,0, channel_to_move]
            new_vector[i,:,channel_to_move] = np.roll(new_vector[i,:,channel_to_move], idel)
            new_vector[i,:idel,channel_to_move] = 0
        new_vector[:,:,channel_to_fix] = vector[:,:,channel_to_fix]


    if delay_time < 0:
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1] - 1):
                    slope = (vector[i, j + 1, channel_to_move] - vector[i, j, channel_to_move]) / time_step
                    new_vector[i, j, channel_to_move] =  vector[i, j, channel_to_move] + slope * res 
            new_vector[i,-1, channel_to_move] = vector[i,-1, channel_to_move]
            if idel <= -1:
                new_vector[i,:,channel_to_move] = np.roll(new_vector[i,:,channel_to_move], idel)
                new_vector[i,idel:,channel_to_move] = vector[i, idel:, channel_to_move]
            else:
                pass
        new_vector[:,:,channel_to_fix] = vector[:,:,channel_to_fix]
    return new_vector

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def extract_signal_window_by_fraction(vector, fraction = 0.2, window_low = 140, window_high = 10):

    new_vector = np.zeros((vector.shape[0], int(window_high + window_low), 2))
    
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
        new_vector[i,:, 0] =  vector[i,low_index_channel0:high_index_channel0, 0]
        new_vector[i,:, 1] =  vector[i,low_index_channel1:high_index_channel1, 1]
        

    return new_vector

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def extract_signal_along_time(vector, time_step, fraction = 0.2, window_low = 140, window_high = 10):

    new_vector = np.zeros((vector.shape[0], int(window_high + window_low), 2))
    time_vector = np.zeros((vector.shape[0], int(window_high + window_low), 2))
    t = np.arange(0, time_step*vector.shape[1], time_step)

    a = 0
    b = 0    
    for i in range(vector.shape[0]):
        # Find indices where the signal in each channel exceeds the fraction threshold
        indices_channel0 = np.where(vector[i,:, 0] >= fraction)[0]
        indices_channel1 = np.where(vector[i,:, 1] >= fraction)[0]
        
        if indices_channel0[0] < indices_channel1[0]:
            index = indices_channel0[0]
            a += 1
        if indices_channel1[0] < indices_channel0[0]:
            index = indices_channel1[0]
            b += 1 
        elif indices_channel0[0] == indices_channel1[0]:
            index = indices_channel0[0]
             
        # Calculate the low and high indices to extraction
        index_low = index - window_low
        index_high = index + window_high

        # Extract cropped waveform and put into new vector
        new_vector[i,:, 0] =  vector[i,index_low:index_high, 0]
        new_vector[i,:, 1] =  vector[i,index_low:index_high, 1]

        time_vector[i,:,0] = t[index_low:index_high]
        time_vector[i,:,1] = t[index_low:index_high]

    return new_vector, time_vector, a, b    

def extract_signal_along_time_singles(vector, time_step, fraction = 0.2, window_low = 140, window_high = 10):

    new_vector = np.zeros((vector.shape[0], int(window_high + window_low)))
    time_vector = np.zeros((vector.shape[0], int(window_high + window_low)))
    t = np.arange(0, time_step*vector.shape[1], time_step)
 
    for i in range(vector.shape[0]):
        
        # Find indices where the signal in each channel exceeds the fraction threshold
        index = np.where(vector[i,:] > fraction)[0][0]
        
        # Calculate the low and high indices to extraction
        index_low = index - window_low
        index_high = index + window_high
        
        # Extract cropped waveform and put into new vector
        new_vector[i,:] =  vector[i,index_low:index_high]
        time_vector[i,:] = t[index_low:index_high]
        

    return new_vector, time_vector

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_and_delay_pulse_pair_along_time(pulse_set, time_vector, delay_time = 1):
    

    INPUT = np.zeros((pulse_set.shape[0], pulse_set.shape[1], 2))
    Time_delayed = np.zeros((time_vector.shape[0], time_vector.shape[1], 2))
    REF = np.zeros((pulse_set.shape[0],), dtype = np.float32)

    NRD0 = np.random.uniform(low = 0, high = delay_time, size = pulse_set.shape[0])
    NRD1 = np.random.uniform(low = 0, high = delay_time, size = pulse_set.shape[0])
    
    INPUT[:,:,0] = pulse_set
    INPUT[:,:,1] = pulse_set

    for i in range(pulse_set.shape[0]):
        
        REF[i] = NRD0[i] - NRD1[i]
        
        Time_delayed[i,:,0] = time_vector[i,:] + NRD0[i] 
        Time_delayed[i,:,1] = time_vector[i,:] + NRD1[i] 
 
    return INPUT, REF, Time_delayed

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def momentos_threshold(vector, time_vector, order = 4):
    
    if  len(np.shape(vector)) == 2:
        Nev, Nt = np.shape(vector)
        MOMENT = np.zeros((Nev, order))
        for i in range(Nev):
                for j in range(order): 
                    W = (time_vector[i,:])**(j)
                    MOMENT[i,j] = np.sum(W*vector[i,:])       
   
    
    if  len(np.shape(vector)) == 3:
        Nev, Nt, Nc = np.shape(vector) # Nev: Núm eventos, Nt: Núm puntos temporales, Nc: Número canales
        MOMENT = np.zeros((Nev, order, Nc))
        for i in range(Nev):
            for j in range(order): # Number of moments used
                for k in range(Nc):
                    W = (time_vector[i,:,k])**(j)
                    MOMENT[i,j,k] = np.sum(W*vector[i,:,k])
    
    return MOMENT

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_dataloaders(input, target, batch_size = 32, shuffle = True):
  dataset = torch.utils.data.TensorDataset(torch.from_numpy(input).float(), torch.from_numpy(np.expand_dims(target, axis = -1)).float())
  dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
  return dataloader

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
