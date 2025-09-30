import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config import (
    device, seed, batch_size, epochs, learning_rate, before, after,
    delay_time, nbins, threshold, DATA_DIR, MODEL_SAVE_DIR, BASE_DIR,
    FIGURES_DIR, model_type, model_name_dec0, model_name_dec1
)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions 
from functions import (create_and_delay_pulse_pair, set_seed, create_dataloaders, 
                       calculate_gaussian_center, plot_gaussian, get_gaussian_params,
                       extract_signal_window_by_fraction)
from Models import ConvolutionalModel,  count_parameters
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

# Delays
moved_pulses_dec0, delays_dec0 = extract_signal_window_by_fraction(train_data[:,:,0], time_step, fraction = threshold, window_low = before, window_high = after)
moved_pulses_dec1, delays_dec1 = extract_signal_window_by_fraction(train_data[:,:,1], time_step, fraction = threshold, window_low = before, window_high = after)

moved_pulses_val_dec0, delays_val_dec0  = extract_signal_window_by_fraction(validation_data[:,:,0], time_step, fraction = threshold, window_low = before, window_high = after)
moved_pulses_val_dec1, delays_val_dec1  = extract_signal_window_by_fraction(validation_data[:,:,1], time_step, fraction = threshold, window_low = before, window_high = after)

moved_pulses_test_dec0, delays_test_dec0 = extract_signal_window_by_fraction(test_data[:,:,0], time_step, fraction = threshold, window_low = before, window_high = after)
moved_pulses_test_dec1, delays_test_dec1 = extract_signal_window_by_fraction(test_data[:,:,1], time_step, fraction = threshold, window_low = before, window_high = after)


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

if model_type == 'CNN':
    model_dec0 = ConvolutionalModel(int(before + after))
    model_dec1 = ConvolutionalModel(int(before + after))
else:
    raise ValueError(f"Unsupported model_type: {model_type}. This routine is for 'CNN' models only.")

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = learning_rate) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = learning_rate) 

# Execute train loop
test_dec0, val_dec0 = train_loop(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, EPOCHS = epochs, model_name = model_name_dec0, model_dir = MODEL_SAVE_DIR, model_type = model_type, figure_dir = FIGURES_DIR, test_tensor = torch.tensor(TEST[:,:,0]).float()) 
test_dec1, val_dec1 = train_loop(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, EPOCHS = epochs, model_name = model_name_dec1, model_dir = MODEL_SAVE_DIR, model_type = model_type, figure_dir = FIGURES_DIR, test_tensor = torch.tensor(TEST[:,:,1]).float())

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF and decompress
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)

size = int(TOF.shape[1]/Theoretical_TOF.shape[0]) # Size of slice
TOF_dict = dataset.get_TOF_slices_train(TOF)

# Calulate Error
centroid_V00 = calculate_gaussian_center(TOF_dict[0], nbins = nbins, limit = 6) 

error_dict = dataset.compute_error(centroid_V00[:,np.newaxis]) # Get error of each position
Error = np.concatenate(list(error_dict.values()), axis = 1)   # COncatenate all positions
MAE = np.mean(Error, axis = 1)
print(MAE[-1])

# Plot
plt.figure(figsize = (8,5))
plt.plot(MAE, label = 'MAE')
plt.title('Results in coincidence')
plt.xlabel('Epochs')
plt.ylabel('Log10')
plt.legend()
plt.show()
