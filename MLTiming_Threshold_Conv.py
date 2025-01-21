import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


from functions import (set_seed, calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, extract_signal_along_time, 
                       create_and_delay_pulse_pair_along_time, create_dataloaders)
from Models import count_parameters, ConvolutionalModel
from Train_loops import  train_loop_convolutional

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

delay_time = 0.5                           # Max delay to training pulses in ns
moments_order = 5                          # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 71                                 # Num bins for all histograms
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
epochs = 50                               # Number of epochs for training
lr = 1e-4                                  # Model learning rate
save = False                               # Save models or not 
fraction = 0.1                             # Fraction to trigger the pulse cropping   
window_low = 14                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger
batch_size = 16
positions = [-0.2, 0.0, 0.2]               # Expected time difference of each position



train_data = np.concatenate((train_data_55, train_data_28, train_data_82), axis = 0)
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train_array, train_time_array, a, b = extract_signal_along_time(train_data, time_step, fraction = fraction, window_low = window_low, window_high = window_high)
val_array, val_time_array, _ , _ = extract_signal_along_time(validation_data,  time_step, fraction = fraction, window_low = window_low, window_high = window_high)
test_array, test_time_array, _ , _ = extract_signal_along_time(test_data,  time_step, fraction = fraction, window_low = window_low, window_high = window_high)

train_dec0, REF_train_dec0, time_train_dec0 = create_and_delay_pulse_pair_along_time(train_array[:,:,0], train_time_array[:,:,0], delay_time = delay_time)
train_dec1, REF_train_dec1, time_train_dec1 = create_and_delay_pulse_pair_along_time(train_array[:,:,1], train_time_array[:,:,1], delay_time = delay_time)

val_dec0, REF_val_dec0, val_time_dec0 = create_and_delay_pulse_pair_along_time(val_array[:,:,0], val_time_array[:,:,0], delay_time = delay_time)
val_dec1, REF_val_dec1, val_time_dec1 = create_and_delay_pulse_pair_along_time(val_array[:,:,1], val_time_array[:,:,1], delay_time = delay_time)

# Concatenate
train_dec0 = train_dec0 * time_train_dec0
train_dec1 = train_dec1 * time_train_dec1

val_dec0 = val_dec0 * val_time_dec0
val_dec1 = val_dec1 * val_time_dec1

TEST = test_array * test_time_array

# Create Dataloaders
train_loader_dec0 = create_dataloaders(train_dec0, REF_train_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = create_dataloaders(train_dec1, REF_train_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0  = create_dataloaders(val_dec0, REF_val_dec0, batch_size = batch_size, shuffle = False)
val_loader_dec1  = create_dataloaders(val_dec1, REF_val_dec1, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model_dec0 = ConvolutionalModel(int(window_low + window_high))
model_dec1 = ConvolutionalModel(int(window_low + window_high))

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr, weight_decay = 1e-5) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr, weight_decay = 1e-5) 

#Execute train loop
loss_dec0, test_dec0, val_loss_dec0, val_dec0  = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:, 0]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/Conv_model_dec0',  save = save) 
loss_dec1, test_dec1, val_loss_dec1, val_dec1  = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:, 1]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/Conv_model_dec1',  save = save)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V00 = TOF[:,:test_data_55.shape[0]] 
TOF_V02 = TOF[:, test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
TOF_V20 = TOF[:, test_data_55.shape[0]  + test_data_28.shape[0]:] 
   
   
# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 
    
error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[0]))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[2]))

Error = np.concatenate((error_V02, error_V20, error_V00),  axis = 1)  

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
plt.hist(test_dec0[-1,:], bins = nbins, alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[-1,:], bins = nbins, alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(133)
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Log Training loss Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Log Training loss Detector 1')
plt.plot(np.log10(val_loss_dec0.astype('float32')), label = 'Log Validation loss Detector 0')
plt.plot(np.log10(val_loss_dec1.astype('float32')), label = 'Log Validation loss Detector 1')
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
