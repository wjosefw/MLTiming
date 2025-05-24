import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config_Gross_Adjustment import (
    device, seed, batch_size, epochs, learning_rate, before, after, save,
    time_step, delay_time, nbins, positions, step_size, Theoretical_TOF, threshold,
    DATA_DIR, MODEL_SAVE_DIR, BASE_DIR
)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions 
from functions import (create_and_delay_pulse_pair, set_seed, create_dataloaders, 
                       calculate_gaussian_center, plot_gaussian, get_gaussian_params,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel,  count_parameters
from Dataset import Datos_LAB_GFN
from Train_loops import train_loop_convolutional

# -------------------------------------------------------------------------
#---------------------------- LOAD DATA -----------------------------------
# -------------------------------------------------------------------------

train_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_train.npz'), mmap_mode = 'r')['data']
validation_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_val.npz'), mmap_mode = 'r')['data']

dataset = Datos_LAB_GFN(data_dir = DATA_DIR, positions = positions, step_size = step_size)
test_data = dataset.load_data()

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

def extract_signal_window_by_fraction(vector, fraction = 0.1, window_low = 140, window_high = 10):
    """
    Extracts a windowed segment of a signal around a threshold crossing point defined by a fraction 
    of the maximum signal amplitude. Also computes delay indices relative to the crossing point.
    """

    # Initialize an array to hold the extracted windows for each signal and list to hold delay values 
    new_vector = np.zeros((vector.shape[0], int(window_high + window_low)))
    delays_list = []

    # Iterate through each signal in the batch
    for i in range(vector.shape[0]):

        # Find the crossing time where signal reaches a certain fraction of its max amplitude
        crossing = calculate_slope_y_intercept(vector[i, :], time_step, threshold=fraction)
        
        # Convert crossing time to index and define window range
        start = int(crossing / time_step) - window_low
        stop = int(crossing / time_step) + window_high
     
        # Extract the windowed portion of the signal
        new_vector[i, :] = vector[i, start:stop]
        
        # Store the negative start index as the delay for alignment
        delays_list.append(-1 * start)
        
    return new_vector, np.array(delays_list)


# Delays
moved_pulses_dec0, delays_dec0 = extract_signal_window_by_fraction(train_data[:,:,0], fraction = threshold, window_low = before, window_high = after)
moved_pulses_dec1, delays_dec1 = extract_signal_window_by_fraction(train_data[:,:,1], fraction = threshold, window_low = before, window_high = after)

moved_pulses_val_dec0, delays_val_dec0  = extract_signal_window_by_fraction(validation_data[:,:,0], fraction = threshold, window_low = before, window_high = after)
moved_pulses_val_dec1, delays_val_dec1  = extract_signal_window_by_fraction(validation_data[:,:,1], fraction = threshold, window_low = before, window_high = after)

moved_pulses_test_dec0, delays_test_dec0 = extract_signal_window_by_fraction(test_data[:,:,0], fraction = threshold, window_low = before, window_high = after)
moved_pulses_test_dec1, delays_test_dec1 = extract_signal_window_by_fraction(test_data[:,:,1], fraction = threshold, window_low = before, window_high = after)


# Train/Validation/Test set
train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(moved_pulses_dec0, time_step, delay_time = delay_time)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(moved_pulses_dec1, time_step, delay_time = delay_time)

val_dec0, REF_val_dec0 = create_and_delay_pulse_pair(moved_pulses_val_dec0, time_step, delay_time = delay_time)
val_dec1, REF_val_dec1 = create_and_delay_pulse_pair(moved_pulses_val_dec1, time_step, delay_time = delay_time)

TEST = np.stack((moved_pulses_test_dec0, moved_pulses_test_dec1), axis = 2)

# Create Dataloaders
train_loader_dec0 = create_dataloaders(train_dec0, REF_train_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = create_dataloaders(train_dec1, REF_train_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0  = create_dataloaders(val_dec0, REF_val_dec0, batch_size =  val_dec0.shape[0], shuffle = False)
val_loader_dec1  = create_dataloaders(val_dec1, REF_val_dec1, batch_size =  val_dec1.shape[0], shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

set_seed(seed)
model_dec0 = ConvolutionalModel(int(before + after))
set_seed(seed)
model_dec1 = ConvolutionalModel(int(before + after))

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = learning_rate) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = learning_rate) 

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec0'),  save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec1'),  save = save)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF and decompress
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)

size = int(TOF.shape[1]/Theoretical_TOF.shape[0]) # Size of slice
TOF_dict = dataset.get_TOF_slices_train(TOF, size)

# Calulate Error
centroid_V00 = calculate_gaussian_center(TOF_dict[0], nbins = nbins, limit = 6) 

error_dict = dataset.compute_error(centroid_V00[:,np.newaxis]) # Get error of each position
Error = np.concatenate(list(error_dict.values()), axis = 1)   # COncatenate all positions
MAE = np.mean(Error, axis = 1)
print(MAE[-1])

# Plot
plt.figure(figsize = (20,5))
plt.subplot(131)
plt.plot(MAE, label = 'MAE')
plt.title('Results in coincidence')
plt.xlabel('Epochs')
plt.ylabel('Log10')
plt.legend()

plt.subplot(132)
plt.hist(test_dec0[-1, :], bins = nbins, range = [-0.5, 1], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[-1, :], bins = nbins, range = [-0.5, 1], alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(133)
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Training loss Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Training loss Detector 1')
plt.plot(np.log10(val_loss_dec0.astype('float32')), label = 'Val loss Detector 0')
plt.plot(np.log10(val_loss_dec1.astype('float32')), label = 'Val loss Detector 1')
plt.title('Losses')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plt.figure(figsize = (16, 6))
for i in range(np.min(positions), np.max(positions) + 1):     
    plot_gaussian(TOF_dict[i][-1,:], centroid_V00[-1], range = 0.6, label = 'pos' + str(i), nbins = nbins)
    params, errors = get_gaussian_params(TOF_dict[i][-1,:], centroid_V00[-1], range = 0.6, nbins = nbins)
    print(f"{i}: CENTROID(ns) = {params[1]:.4f} +/- {errors[2]:.5f}  FWHM(ns) = {params[2]:.4f} +/- {errors[3]:.5f}")

print('')
plt.legend()
plt.xlabel(r'\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()
