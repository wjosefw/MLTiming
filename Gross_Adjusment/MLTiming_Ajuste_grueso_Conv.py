import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config_Gross_Adjustment import (
    device, seed, batch_size, epochs, learning_rate, Num_Neurons, before, after, save,
    delay_time, nbins, threshold, DATA_DIR, MODEL_SAVE_DIR, 
    REF_PULSE_SAVE_DIR, BASE_DIR)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions 
from functions import (move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, calculate_gaussian_center, 
                       plot_gaussian, get_gaussian_params, get_mean_pulse_from_set,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel,  count_parameters, MLP_Torch
from Dataset import Datos_LAB_GFN
from Train_loops import train_loop

# -------------------------------------------------------------------------
#---------------------------- LOAD DATA -----------------------------------
# -------------------------------------------------------------------------

dataset = Datos_LAB_GFN(data_dir = DATA_DIR)

train_data = dataset.load_train_data()
validation_data = dataset.load_val_data()
test_data = dataset.load_test_data()

time_step, positions, Theoretical_TOF = dataset.load_params() # Load data parameters

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

mean_pulse_dec0 = get_mean_pulse_from_set(train_data, channel = 0)
mean_pulse_dec1 = get_mean_pulse_from_set(train_data, channel = 1)

np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, "reference_pulse_dec0.npz"), data = mean_pulse_dec0)
np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, "reference_pulse_dec1.npz"), data = mean_pulse_dec1)

# Get start and stop
crossing_dec0 = calculate_slope_y_intercept(mean_pulse_dec0, time_step, threshold = threshold)
crossing_dec1 = calculate_slope_y_intercept(mean_pulse_dec1, time_step, threshold = threshold)

start_dec0 = int(crossing_dec0/time_step) - before
start_dec1 = int(crossing_dec1/time_step) - before
stop_dec0 = int(crossing_dec0/time_step) + after
stop_dec1 = int(crossing_dec1/time_step) + after

# Delays
delays_dec0, moved_pulses_dec0 = move_to_reference(mean_pulse_dec0, train_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_dec1, moved_pulses_dec1 = move_to_reference(mean_pulse_dec1, train_data, start = start_dec1, stop = stop_dec1, channel = 1)

delays_val_dec0, moved_pulses_val_dec0 = move_to_reference(mean_pulse_dec0, validation_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_val_dec1, moved_pulses_val_dec1 = move_to_reference(mean_pulse_dec1, validation_data, start = start_dec1, stop = stop_dec1, channel = 1)

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start_dec1, stop = stop_dec1, channel = 1)

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
model_dec0 = ConvolutionalModel(int(stop_dec0-start_dec0))
set_seed(seed)
model_dec1 = ConvolutionalModel(int(stop_dec1-start_dec1))

#set_seed(seed)
#model_dec0 = MLP_Torch(NM = int(stop_dec0-start_dec0), NN = Num_Neurons, STD_INIT = 0.5)
#set_seed(seed)
#model_dec1 = MLP_Torch(NM = int(stop_dec1-start_dec1), NN = Num_Neurons, STD_INIT = 0.5)

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = learning_rate) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = learning_rate) 

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec0'),  save = save, model_type = 'CNN', test_tensor = torch.tensor(TEST[:,:,0]).float()) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec1'),  save = save, model_type = 'CNN', test_tensor = torch.tensor(TEST[:,:,1]).float())

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF and decompress
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)
TOF_dict = dataset.get_TOF_slices_train(TOF) # Cut in slices per each position

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

# -------------------------------------------------------------------------
# -------------------- SAVE RESULTS OVER EPOCHS ---------------------------
# -------------------------------------------------------------------------

#idx = 0
#val_dec0 = val_dec0[idx:,:,:] 
#val_dec1 = val_dec1[idx:,:,:]
#MAE = MAE[idx:]
#
## Plot MAE_singles vs MAE_coincidences
#err_val_dec0 = abs(val_dec0[:,:,0] - val_dec0[:,:,1] - REF_val_dec0[np.newaxis,:])
#err_val_dec1 = abs(val_dec1[:,:,0] - val_dec1[:,:,1] - REF_val_dec1[np.newaxis,:])
#mean_err_val_dec0 = np.mean(err_val_dec0, axis = 1)
#mean_err_val_dec1 = np.mean(err_val_dec1, axis = 1)
#np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, 'mean_err_val_dec0_Na22.npz'), data = mean_err_val_dec0)
#np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, 'mean_err_val_dec1_Na22.npz'), data = mean_err_val_dec1)
#np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, 'MAE_Na22.npz'), data = MAE)
#
#
#avg_bias = np.zeros((TOF_dict[0].shape[0],))
#CTR = np.zeros((TOF_dict[0].shape[0],))
#centroid_0 = calculate_gaussian_center(TOF_dict[0], nbins = nbins, limit = 3) 
#for i in range(TOF_dict[0].shape[0]):
#
#    resolution_dict = {} # Initialize dictionaries
#    centroids_dict = {} 
#    error_dict = {} 
#
#    for j in range(np.min(positions), np.max(positions) + 1):  
#        params, errors = get_gaussian_params(TOF_dict[j][i,idx:], centroid_0[i], range = 0.6, nbins = nbins)
#        resolution_dict[j] = params[2]
#        centroids_dict[j] = params[1]
#    
#    centroids = list(centroids_dict.values())   
#    avg_bias[i] = np.mean(abs(centroids - Theoretical_TOF))  
#    CTR[i] = np.mean(list(resolution_dict.values()))  
#
#np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, 'ctr.npz'), data = CTR)
#np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, 'avg_bias.npz'), data = avg_bias)