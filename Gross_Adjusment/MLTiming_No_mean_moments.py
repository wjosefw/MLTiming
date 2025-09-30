import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config import (
    device, seed, batch_size, epochs, learning_rate, before, after,
    delay_time, nbins, threshold, DATA_DIR, MODEL_SAVE_DIR, BASE_DIR,
    FIGURES_DIR, model_type, model_name_dec0, model_name_dec1, 
    moments_order, normalization_method, Num_Neurons, architecture
)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions 
from functions import (create_and_delay_pulse_pair, set_seed, create_dataloaders, 
                       calculate_gaussian_center, plot_gaussian, get_gaussian_params,
                       extract_signal_window_by_fraction, normalize_given_params, momentos,
                       normalize)
from Models import count_parameters, MLP_Torch
from Dataset import Datos_LAB_GFN
from Train_loops import train_loop
from efficient_kan.src.efficient_kan import KAN

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

# Calculate moments 
M_Train_dec0 = momentos(train_dec0, order = moments_order) 
M_Train_dec1 = momentos(train_dec1, order = moments_order) 

M_Val_dec0 = momentos(val_dec0, order = moments_order) 
M_Val_dec1 = momentos(val_dec1, order = moments_order) 

M_Test = momentos(TEST, order = moments_order)

# Normalize moments
M_Train_dec0, params_dec0 =  normalize(M_Train_dec0, method = normalization_method)
M_Train_dec1, params_dec1 =  normalize(M_Train_dec1, method = normalization_method)

M_Val_dec0_channel0 =  normalize_given_params(M_Val_dec0, params_dec0, channel = 0, method = normalization_method)
M_Val_dec0_channel1 =  normalize_given_params(M_Val_dec0, params_dec0, channel = 1, method = normalization_method)
M_Val_dec0 = np.stack((M_Val_dec0_channel0, M_Val_dec0_channel1), axis = -1)

M_Val_dec1_channel0 =  normalize_given_params(M_Val_dec1, params_dec1, channel = 0, method = normalization_method)
M_Val_dec1_channel1 =  normalize_given_params(M_Val_dec1, params_dec1, channel = 1, method = normalization_method)
M_Val_dec1 = np.stack((M_Val_dec1_channel0, M_Val_dec1_channel1), axis = -1)

M_Test_norm_dec0 = normalize_given_params(M_Test, params_dec0, channel = 0, method = normalization_method)
M_Test_norm_dec1 = normalize_given_params(M_Test, params_dec1, channel = 1, method = normalization_method)
M_Test = np.stack((M_Test_norm_dec0, M_Test_norm_dec1), axis = -1)
M_Test = torch.tensor(M_Test, dtype = torch.float32, device = device)

# Create Dataloaders
train_loader_dec0 = create_dataloaders(M_Train_dec0, REF_train_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = create_dataloaders(M_Train_dec1, REF_train_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0  = create_dataloaders(M_Val_dec0, REF_val_dec0, batch_size = M_Val_dec0.shape[0], shuffle = False)
val_loader_dec1  = create_dataloaders(M_Val_dec1, REF_val_dec1, batch_size = M_Val_dec1.shape[0], shuffle = False)

# Print information 
print("Normalization parameters detector 0:", params_dec0)
print("Normalization parameters detector 1:", params_dec1)


# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

if model_type == 'MLP':
    model_dec0 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
    model_dec1 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
elif model_type == 'KAN':
    model_dec0 = KAN(architecture)
    model_dec1 = KAN(architecture)
else:
    raise ValueError(f"Unsupported model_type: {model_type}. This routine is for 'MLP' or 'KAN' models only.")

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = learning_rate) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = learning_rate) 

# Execute train loop
test_dec0, val_dec0 = train_loop(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, EPOCHS = epochs, model_name = model_name_dec0, model_dir = MODEL_SAVE_DIR, model_type = model_type, figure_dir = FIGURES_DIR, test_tensor = M_Test[:,:,0]) 
test_dec1, val_dec1 = train_loop(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, EPOCHS = epochs, model_name = model_name_dec1, model_dir = MODEL_SAVE_DIR, model_type = model_type, figure_dir = FIGURES_DIR, test_tensor = M_Test[:,:,1])

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