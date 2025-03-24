import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config_Fixed_Window import (device, delay_time, time_step, nbins, 
                                 seed, epochs, lr, batch_size, save, 
                                 positions, start, stop, Theoretical_TOF,
                                 BASE_DIR, MODEL_SAVE_DIR, DATA_DIR, step_size)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions
from functions import (create_and_delay_pulse_pair, calculate_gaussian_center, 
                       plot_gaussian, get_gaussian_params, set_seed,
                       create_dataloaders)
from Models import ConvolutionalModel,  count_parameters
from Dataset import Datos_LAB_GFN
from Train_loops import train_loop_convolutional

# -------------------------------------------------------------------------
#---------------------------- LOAD DATA -----------------------------------
# -------------------------------------------------------------------------

dataset = Datos_LAB_GFN(data_dir = DATA_DIR, positions = positions, step_size = step_size)

train_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_train.npz'), mmap_mode = 'r')['data']
validation_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_val.npz'), mmap_mode = 'r')['data']
test_data = dataset.load_data()

train_data = train_data[:,start:stop,:]
validation_data = validation_data[:,start:stop,:] 
test_data = test_data[:,start:stop,:]

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(train_data[:,:,0], time_step, delay_time = delay_time)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(train_data[:,:,1], time_step, delay_time = delay_time)

val_dec0, REF_val_dec0 = create_and_delay_pulse_pair(validation_data[:,:,0], time_step, delay_time = delay_time)
val_dec1, REF_val_dec1 = create_and_delay_pulse_pair(validation_data[:,:,1], time_step, delay_time = delay_time)

# Create Dataloaders
train_loader_dec0 = create_dataloaders(train_dec0, REF_train_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = create_dataloaders(train_dec1, REF_train_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0  = create_dataloaders(val_dec0, REF_val_dec0, batch_size =  val_dec0.shape[0], shuffle = False)
val_loader_dec1  = create_dataloaders(val_dec1, REF_val_dec1, batch_size =  val_dec1.shape[0], shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

set_seed(seed)
model_dec0 = ConvolutionalModel(int(stop-start))
set_seed(seed)
model_dec1 = ConvolutionalModel(int(stop-start))

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr) 

#Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(test_data[:,:,0]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'Fixed_Window_model_dec0'),  save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(test_data[:,:,1]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'Fixed_Window_model_dec1'),  save = save)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF and decompress
TOF = test_dec0 - test_dec1 

size = int(TOF.shape[1]/Theoretical_TOF.shape[0]) # Size of slice
TOF_dict = dataset.get_TOF_slices_train(TOF, size)

# Calulate Error
centroid_V00 = calculate_gaussian_center(TOF_dict[0], nbins = nbins, limit = 6) 

error_dict = dataset.compute_error(centroid_V00[:,np.newaxis]) # Get error of each position
Error = np.concatenate(list(error_dict.values()), axis = 1)   # Concatenate all positions
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
plt.hist(test_dec0[-1,:], bins = nbins, range = [-1, 3], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[-1,:], bins = nbins, range = [-1, 3], alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(133)
plt.plot(loss_dec0, label = 'Log Training loss Detector 0')
plt.plot(loss_dec1, label = 'Log Training loss Detector 1')
plt.plot(val_loss_dec0, label = 'Log Validation loss Detector 0')
plt.plot(val_loss_dec1, label = 'Log Validation loss Detector 1')
plt.ylabel('Logarithmic losses')
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