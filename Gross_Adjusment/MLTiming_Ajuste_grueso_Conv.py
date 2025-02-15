import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config_Gross_Adjustment import (
    device, seed, batch_size, epochs, learning_rate, Num_Neurons, before, after, save,
    time_step, delay_time, nbins, positions, DATA_DIR, MODEL_SAVE_DIR, REF_PULSE_SAVE_DIR, 
    BASE_DIR
)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions 
from functions import (move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, calculate_gaussian_center, 
                       plot_gaussian, get_gaussian_params, get_mean_pulse_from_set,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel,  count_parameters, MLP_Torch
from Train_loops import train_loop_convolutional, train_loop_MLP

# Load data
data0_train = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_train.npz'), mmap_mode='r')['data']
data0_val = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_val.npz'), mmap_mode='r')['data']

test_data_dict = {}
for i in range(-5, 6):  
    filename = f"Na22_norm_pos{i}_test.npz" if i >= 0 else f"Na22_norm_pos_min_{abs(i)}_test.npz"
    test_data_dict[i] = np.load(DATA_DIR / filename, mmap_mode = "r")["data"]

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = data0_train
validation_data = data0_val
test_data = np.concatenate((test_data_dict[-5], test_data_dict[-4], test_data_dict[-3], 
                            test_data_dict[-2], test_data_dict[-1], test_data_dict[0],  
                            test_data_dict[1], test_data_dict[2], test_data_dict[3],
                            test_data_dict[4], test_data_dict[5]), axis = 0)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

mean_pulse_dec0 = get_mean_pulse_from_set(train_data, channel = 0)
mean_pulse_dec1 = get_mean_pulse_from_set(train_data, channel = 1)

np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, "reference_pulse_dec0.npz"), data=mean_pulse_dec0)
np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, "reference_pulse_dec1.npz"), data=mean_pulse_dec1)

# Get start and stop
crossing_dec0 = calculate_slope_y_intercept(mean_pulse_dec0, time_step, threshold = 0.1)
crossing_dec1 = calculate_slope_y_intercept(mean_pulse_dec1, time_step, threshold = 0.1)

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

set_seed(42)
model_dec0 = ConvolutionalModel(int(stop_dec0-start_dec0))
set_seed(42)
model_dec1 = ConvolutionalModel(int(stop_dec1-start_dec1))

#set_seed(42)
#model_dec0 = MLP_Torch(NM = int(stop_dec0-start_dec0), NN = Num_Neurons, STD_INIT = 0.5)
#set_seed(42)
#model_dec1 = MLP_Torch(NM = int(stop_dec1-start_dec1), NN = Num_Neurons, STD_INIT = 0.5)

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = learning_rate) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = learning_rate) 

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec0'),  save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec1'),  save = save)
#loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_MLP(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_AG_model_dec0'), save = save) 
#loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_MLP(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_AG_model_dec0'), save = save)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)

size = int(TOF.shape[1]/positions.shape[0])
TOF_dict = {}  
for i in range(-5, 6):  
    TOF_dict[i] = TOF[:, (i + 5) * size : (i + 6) * size]  # Assign slices dynamically

# Calulate Error
centroid_V00 = calculate_gaussian_center(TOF_dict[0][np.newaxis,:], nbins = nbins, limit = 6) 
error_dict = {} 
for i in range(-5, 6):  
    error_dict[i] = abs(TOF_dict[i] - centroid_V00[:, np.newaxis] - positions[i + 5])  # Compute error per position

Error = np.concatenate(list(error_dict.values()), axis = 1)   
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
for i in range(-5, 6):  
    plot_gaussian(TOF_dict[i][-1,:], centroid_V00, range = 0.6, label = 'pos' + str(i), nbins = nbins)
    params, errors = get_gaussian_params(TOF_dict[i][-1,:], centroid_V00, range = 0.6, nbins = nbins)
    print(f"{i}: CENTROID(ns) = {params[1]:.4f} +/- {errors[2]:.5f}  FWHM(ns) = {params[2]:.4f} +/- {errors[3]:.5f}")

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

# -------------------------------------------------------------------------
# -------------------- SAVE RESULTS OVER EPOCHS ---------------------------
# -------------------------------------------------------------------------

idx = 0
TOF_0 = TOF_dict[0][idx:]
TOF_1 = TOF_dict[1][idx:]
TOF_2 = TOF_dict[2][idx:]
TOF_3 = TOF_dict[3][idx:]
TOF_4 = TOF_dict[4][idx:]
TOF_5 = TOF_dict[5][idx:]
TOF_min_1 = TOF_dict[-1][idx:]
TOF_min_2 = TOF_dict[-2][idx:]
TOF_min_3 = TOF_dict[-3][idx:]
TOF_min_4 = TOF_dict[-4][idx:]
TOF_min_5 = TOF_dict[-5][idx:]

val_dec0 = val_dec0[idx:,:,:] 
val_dec1 = val_dec1[idx:,:,:]
MAE = MAE[idx:]

# Plot MAE_singles vs MAE_coincidences
err_val_dec0 = abs(val_dec0[:,:,0] - val_dec0[:,:,1] - REF_val_dec0[np.newaxis,:])
err_val_dec1 = abs(val_dec1[:,:,0] - val_dec1[:,:,1] - REF_val_dec1[np.newaxis,:])
mean_err_val_dec0 = np.mean(err_val_dec0, axis = 1)
mean_err_val_dec1 = np.mean(err_val_dec1, axis = 1)
np.savez_compressed('../predictions/mean_err_val_dec0_Na22.npz', data = mean_err_val_dec0)
np.savez_compressed('../predictions/mean_err_val_dec1_Na22.npz', data = mean_err_val_dec1)
np.savez_compressed('../predictions/MAE_Na22.npz', data = MAE)

CTR = []
avg_bias = []
centroid_0 = calculate_gaussian_center(TOF_0, nbins = nbins, limit = 3) 
for i in range(TOF_0.shape[0]):
    params_0, errors_0 = get_gaussian_params(TOF_0[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_1, errors_1 = get_gaussian_params(TOF_1[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_2, errors_2 = get_gaussian_params(TOF_2[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_3, errors_3 = get_gaussian_params(TOF_3[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_4, errors_4 = get_gaussian_params(TOF_4[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_5, errors_5 = get_gaussian_params(TOF_5[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_min_1, errors_min_1 = get_gaussian_params(TOF_min_1[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_min_2, errors_min_2 = get_gaussian_params(TOF_min_2[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_min_3, errors_min_3 = get_gaussian_params(TOF_min_3[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_min_4, errors_min_4 = get_gaussian_params(TOF_min_4[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    params_min_5, errors_min_5 = get_gaussian_params(TOF_min_5[i,:], centroid_0[i], range = 0.6, nbins = nbins)
    
    CTR.append(np.mean([params_0[2], params_1[2], params_2[2], 
                        params_3[2], params_4[2], params_5[2],  
                        params_min_1[2],  params_min_2[2], params_min_3[2],
                        params_min_4[2],  params_min_5[2]]))
    
    error_min_5_centroid = abs((params_min_5[1] - positions[0]))
    error_min_4_centroid = abs((params_min_4[1] - positions[1]))
    error_min_3_centroid = abs((params_min_3[1] - positions[2]))
    error_min_2_centroid = abs((params_min_2[1] - positions[3]))
    error_min_1_centroid = abs((params_min_1[1] - positions[4]))

    error_0_centroid = abs((params_0[1] - positions[5]))
    error_1_centroid = abs((params_1[1] - positions[6]))
    error_2_centroid = abs((params_2[1] - positions[7]))
    error_3_centroid = abs((params_3[1] - positions[8]))
    error_4_centroid = abs((params_4[1] - positions[9]))
    error_5_centroid = abs((params_5[1] - positions[10]))


    avg_bias.append(np.mean([error_0_centroid, error_1_centroid, error_2_centroid, 
                             error_3_centroid, error_4_centroid, error_5_centroid,
                             error_min_1_centroid,  error_min_2_centroid, error_min_3_centroid,
                             error_min_4_centroid,  error_min_5_centroid]))
   
np.savez_compressed('../predictions/ctr.npz', data = np.array(CTR))
np.savez_compressed('../predictions/avg_bias.npz', data = np.array(avg_bias))
