import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


from functions import (momentos_threshold, set_seed, calculate_gaussian_center, 
                       normalize, normalize_given_params, plot_gaussian, get_gaussian_params, 
                       extract_signal_along_time_singles, create_and_delay_pulse_pair_along_time)
from Train_loops import train_loop_KAN


# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'

train_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_train.npz'))['data']
train_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_train.npz'))['data']
train_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_train.npz'))['data']

validation_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_val.npz'))['data']
validation_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_val.npz'))['data']
validation_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_val.npz'))['data']

test_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_test.npz'))['data']
test_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_test.npz'))['data']
test_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_test.npz'))['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_time = 1                             # Max delay to training pulses in ns
moments_order = 7                          # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 71                                 # Num bins for all histograms
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
epochs = 500                               # Number of epochs for training
lr = 1e-3                                  # Model learning rate
batch_size = 16                            # batch size used for training
save = False                               # Save models or not
architecture = [moments_order, 3, 1, 1]   
fraction = 0.1                             # Fraction to trigger the pulse cropping   
window_low = 14                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger
positions = np.array([-0.2, 0.0, 0.2])

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = np.concatenate((train_data_55, train_data_28, train_data_82), axis = 0)
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------


# Extract pulses
train_array_dec0, train_time_array_dec0 = extract_signal_along_time_singles(train_data[:,:100,0], time_step, fraction = fraction, window_low = window_low, window_high = window_high)
train_array_dec1, train_time_array_dec1 = extract_signal_along_time_singles(train_data[:,:100,1], time_step, fraction = fraction, window_low = window_low, window_high = window_high)

val_array_dec0, val_time_array_dec0 = extract_signal_along_time_singles(validation_data[:,:100,0], time_step, fraction = fraction, window_low = window_low, window_high = window_high)
val_array_dec1, val_time_array_dec1 = extract_signal_along_time_singles(validation_data[:,:100,1], time_step, fraction = fraction, window_low = window_low, window_high = window_high)

test_array_dec0, test_time_array_dec0  = extract_signal_along_time_singles(test_data[:,:100,0], time_step, fraction = fraction, window_low = window_low, window_high = window_high)
test_array_dec1, test_time_array_dec1  = extract_signal_along_time_singles(test_data[:,:100,1], time_step, fraction = fraction, window_low = window_low, window_high = window_high)

# Stack both detectors
train_array = np.stack((train_array_dec0, train_array_dec1), axis = -1)
train_time_array = np.stack((train_time_array_dec0, train_time_array_dec1), axis = -1)

val_array = np.stack((val_array_dec0, val_array_dec1), axis = -1)
val_time_array = np.stack((val_time_array_dec0, val_time_array_dec1), axis = -1)

test_array = np.stack((test_array_dec0, test_array_dec1), axis = -1)
test_time_array = np.stack((test_time_array_dec0, test_time_array_dec1), axis = -1)


# Create virtual coincidences
train_dec0, REF_train_dec0, time_train_dec0 = create_and_delay_pulse_pair_along_time(train_array[:,:,0], train_time_array[:,:,0], delay_time = delay_time)
train_dec1, REF_train_dec1, time_train_dec1 = create_and_delay_pulse_pair_along_time(train_array[:,:,1], train_time_array[:,:,1], delay_time = delay_time)

val_dec0, REF_val_dec0, val_time_dec0 = create_and_delay_pulse_pair_along_time(val_array[:,:,0], val_time_array[:,:,0], delay_time = delay_time)
val_dec1, REF_val_dec1, val_time_dec1 = create_and_delay_pulse_pair_along_time(val_array[:,:,1], val_time_array[:,:,1], delay_time = delay_time)

TEST = test_array 


# Calculate moments
M_Train_dec0 = momentos_threshold(train_dec0, time_train_dec0, order = moments_order) 
M_Train_dec1 = momentos_threshold(train_dec1, time_train_dec1, order = moments_order) 

M_Val_dec0 = momentos_threshold(val_dec0, val_time_dec0, order = moments_order) 
M_Val_dec1 = momentos_threshold(val_dec1, val_time_dec1, order = moments_order)

M_Test = momentos_threshold(TEST, test_time_array, order = moments_order)

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

# Print information 
print("Normalization parameters detector 0:", params_dec0)
print("Normalization parameters detector 1:", params_dec1)

# Create Dataset
train_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(M_Train_dec0).float(), torch.from_numpy(np.expand_dims(REF_train_dec0, axis = -1)).float())
train_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(M_Train_dec1).float(), torch.from_numpy(np.expand_dims(REF_train_dec1, axis = -1)).float())

val_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(M_Val_dec0).float(), torch.from_numpy(np.expand_dims(REF_val_dec0, axis = -1)).float())
val_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(M_Val_dec1).float(), torch.from_numpy(np.expand_dims(REF_val_dec1, axis = -1)).float())

# Create DataLoaders
train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = len(val_dataset_dec0), shuffle = False)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = len(val_dataset_dec1), shuffle = False)



# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

# Initialize model
set_seed(42)
model_dec0 = KAN(architecture)
set_seed(42)
model_dec1 = KAN(architecture)

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr)  

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(M_Test[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec0', save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(M_Test[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec1', save = save)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V00 = TOF[:,:test_data_55.shape[0]] 
TOF_V02 = TOF[:,test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
TOF_V20 = TOF[:,test_data_55.shape[0] + test_data_28.shape[0]:] 

        
# Calculate centered position 'centroid'
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[0]))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[2]))

Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1)
  
# Print MAE
MAE = np.mean(Error, axis = 1)
print(MAE[-1])

# Plot
plt.figure(figsize = (20,5))
plt.subplot(131)
plt.plot(np.log10(MAE.astype('float64')), label = 'MAE')
plt.title('Results in coincidence')
plt.xlabel('Epochs')
plt.ylabel('Log10')
plt.legend()

plt.subplot(132)
plt.hist(test_dec0[-1,:], bins = nbins, alpha = 0.5, range = [-1, 1.5], label = 'Detector 0');
plt.hist(test_dec1[-1,:], bins = nbins, alpha = 0.5, range = [-1, 1.5], label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(133)
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Train loss Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Train loss Detector 1')
plt.plot(np.log10(val_loss_dec0.astype('float32')), label = 'Val loss Detector 0')
plt.plot(np.log10(val_loss_dec1.astype('float32')), label = 'Val loss Detector 1')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Histogram and gaussian fit 
plot_gaussian(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)

params_V02, errors_V02 = get_gaussian_params(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)

print("V20: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)')
plt.ylabel('Counts')
plt.show()
