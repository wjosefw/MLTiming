import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config_Gross_Adjustment import (
    device, seed, batch_size, epochs, learning_rate, Num_Neurons, before, after, save, 
    moments_order, time_step, delay_time, nbins, positions, Theoretical_TOF, 
    normalization_method, DATA_DIR, MODEL_SAVE_DIR, REF_PULSE_SAVE_DIR, BASE_DIR
)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions 
from functions import (momentos, move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, calculate_gaussian_center, 
                       normalize, normalize_given_params, plot_gaussian, 
                       get_gaussian_params, get_mean_pulse_from_set,
                       calculate_slope_y_intercept)
from Models import MLP_Torch,  count_parameters
from Train_loops import train_loop_KAN, train_loop_MLP
from Dataset import Datos_LAB_GFN
from efficient_kan.src.efficient_kan import KAN

# -------------------------------------------------------------------------
#---------------------------- LOAD DATA -----------------------------------
# -------------------------------------------------------------------------

train_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_train.npz'), mmap_mode = 'r')['data']
validation_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_val.npz'), mmap_mode = 'r')['data']

dataset = Datos_LAB_GFN(data_dir = DATA_DIR)
test_data = dataset.load_data()

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

mean_pulse_dec0 = get_mean_pulse_from_set(train_data, channel = 0)
mean_pulse_dec1 = get_mean_pulse_from_set(train_data, channel = 1)

#np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, "reference_pulse_dec0.npz"), data = mean_pulse_dec0)
#np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, "reference_pulse_dec1.npz"), data = mean_pulse_dec1)

# Get start and stop
crossing_dec0 = calculate_slope_y_intercept(mean_pulse_dec0, time_step, threshold = 0.1)
crossing_dec1 = calculate_slope_y_intercept(mean_pulse_dec1, time_step, threshold = 0.1)

start_dec0 = int(crossing_dec0/time_step) - before
start_dec1 = int(crossing_dec1/time_step) - before
stop_dec0 = int(crossing_dec0/time_step) + after
stop_dec1 = int(crossing_dec1/time_step) + after

# Train/Validation/Test set
delays_dec0, moved_pulses_dec0 = move_to_reference(mean_pulse_dec0, train_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_dec1, moved_pulses_dec1 = move_to_reference(mean_pulse_dec1, train_data, start = start_dec1, stop = stop_dec1, channel = 1)

delays_val_dec0, moved_pulses_val_dec0 = move_to_reference(mean_pulse_dec0, validation_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_val_dec1, moved_pulses_val_dec1 = move_to_reference(mean_pulse_dec1, validation_data, start = start_dec1, stop = stop_dec1, channel = 1)

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start_dec1, stop = stop_dec1, channel = 1)

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

#model_dec0 = KAN(architecture)
#model_dec1 = KAN(architecture)
model_dec0 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
model_dec1 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
                  
print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = learning_rate) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = learning_rate)  

# Execute train loop
#loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, M_Test[:,:,0], EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, KAN_AG_model_dec0), save = save) 
#loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, M_Test[:,:,1], EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, KAN_AG_model_dec1), save = save)
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_MLP(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, M_Test[:,:,0], EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'MLP_AG_model_dec0'), save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_MLP(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, M_Test[:,:,1], EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR,'MLP_AG_model_dec1'), save = save)

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
plt.figure(figsize = (16,6))
for i in range(np.min(positions), np.max(positions) + 1):  
    plot_gaussian(TOF_dict[i][-1,:], centroid_V00[-1], range = 0.6, label = 'pos' + str(i), nbins = nbins)
    params, errors = get_gaussian_params(TOF_dict[i][-1,:], centroid_V00[-1], range = 0.6, nbins = nbins)
    print(f"{i}: CENTROID(ns) = {params[1]:.4f} +/- {errors[2]:.5f}  FWHM(ns) = {params[2]:.4f} +/- {errors[3]:.5f}")

print('')
plt.legend()
plt.xlabel(r'\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

## Combine the two numbers
#num = f"{sys.argv[1]}{sys.argv[2]}"
# 
##Your existing variables
#FWHM = np.array([params_min_5[2], params_min_4[2], params_min_3[2],
#                     params_min_2[2], params_min_1[2],  
#                     params_0[2], params_1[2], params_2[2],
#                     params_3[2], params_4[2], params_5[2]]) # ps
#                 
#
#FWHM_err =  np.array([errors_min_5[3], errors_min_4[3], errors_min_3[3],
#                      errors_min_2[3], errors_min_1[3],  
#                      errors_0[3], errors_1[3], errors_2[3],
#                      errors_3[3], errors_4[3], errors_5[3]]) # ps
#
#                      
#centroid = np.array([params_min_5[1], params_min_4[1], params_min_3[1],
#                     params_min_2[1], params_min_1[1],  
#                     params_0[1], params_1[1], params_2[1],
#                     params_3[1], params_4[1], params_5[1]]) # ps
#
#centroid_err = np.array([errors_min_5[2], errors_min_4[2], errors_min_3[2],
#                         errors_min_2[2], errors_min_1[2],  
#                         errors_0[2], errors_1[2], errors_2[2],
#                         errors_3[2], errors_4[2], errors_5[2]]) # ps
#
#
## Multiply by 1000
#FWHM = FWHM * 1000
#FWHM_err = FWHM_err * 1000
#centroid = centroid * 1000
#centroid_err = centroid_err * 1000
#
#with open('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/results_FS.txt', 'a') as file:
#    file.write(f"FWHM_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM)}])  # ps\n")
#    file.write(f"FWHM_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM_err)}]) \n")
#    file.write(f"centroid_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid)}])  \n")
#    file.write(f"centroid_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid_err)}])  \n")
#    file.write(f"MAE_{num} = {MAE[-1]:.7f}  \n")  # Write MAE with one decima
#    file.write(f"mean_FWHM_{num} = np.mean(FWHM_{num})\n")
#    file.write(f"mean_FWHM_err_{num} = np.mean(FWHM_err_{num})\n")
#    file.write(f"mean_bias_{num} = np.mean(abs(centroid_{num} - Theoretical_TOF))\n")
#    file.write("\n")  # Add a new line for better separation
#    file.flush()
#
#del params_0, params_1, params_2, params_3, params_4, params_5
#del params_min_1, params_min_2, params_min_3, params_min_4, params_min_5
#del errors_0, errors_1, errors_2, errors_3, errors_4, errors_5
#del errors_min_1, errors_min_2, errors_min_3, errors_min_4, errors_min_5
#
#gc.collect()  
#torch.cuda.empty_cache()

