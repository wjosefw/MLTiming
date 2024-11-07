import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


from functions import (create_position, set_seed, calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, continuous_delay, extract_signal_along_time, 
                       create_and_delay_pulse_pair_along_time)
from Models import train_loop_convolutional, count_parameters, ConvolutionalModel_Threshold


# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'

train_data = np.load(os.path.join(dir,'Na22_train.npz'))['data']
val_data = np.load(os.path.join(dir, 'Na22_val.npz'))['data']
test_data = np.load(os.path.join(dir, 'Na22_test_val.npz'))['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_time = 0.5                           # Max delay to training pulses in ns
moments_order = 5                          # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 151                                # Num bins for all histograms
t_shift = 1                                # Time steps to move for the new positions
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
epochs = 500                               # Number of epochs for training
lr = 1e-3                                  # Model learning rate
total_time = time_step*train_data.shape[1]
save = False                               # Save models or not
architecture = [moments_order, 5, 1, 1]   
fraction = 0.1                             # Fraction to trigger the pulse cropping   
window_low = 14                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger


# -------------------------------------------------------------------------
#----------------------- ALIGN PULSES -------------------------------------
# -------------------------------------------------------------------------

align_time = 0.56
new_train = continuous_delay(train_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)
new_val = continuous_delay(val_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)
new_test = continuous_delay(test_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train_array, train_time_array, a, b = extract_signal_along_time(new_train, fraction = fraction, window_low = window_low, window_high = window_high)
val_array, val_time_array, _ , _ = extract_signal_along_time(new_val, fraction = fraction, window_low = window_low, window_high = window_high)
test_array, test_time_array, _ , _ = extract_signal_along_time(new_test, fraction = fraction, window_low = window_low, window_high = window_high)

train_dec0, REF_train_dec0, time_train_dec0 = create_and_delay_pulse_pair_along_time(train_array[:,:,0], train_time_array[:,:,0], time_step, total_time, delay_time = delay_time)
train_dec1, REF_train_dec1, time_train_dec1 = create_and_delay_pulse_pair_along_time(train_array[:,:,1], train_time_array[:,:,1], time_step, total_time, delay_time = delay_time)

val_dec0, REF_val_dec0, val_time_dec0 = create_and_delay_pulse_pair_along_time(val_array[:,:,0], val_time_array[:,:,0], time_step, total_time, delay_time = delay_time)
val_dec1, REF_val_dec1, val_time_dec1 = create_and_delay_pulse_pair_along_time(val_array[:,:,1], val_time_array[:,:,1], time_step, total_time, delay_time = delay_time)

#create positions
TEST_00 = test_array 
TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift)
TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift)
TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift))
TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift))

#move time arrays for positions
test_time_array_02 = np.zeros_like(test_time_array)
test_time_array_04 = np.zeros_like(test_time_array)
test_time_array_20 = np.zeros_like(test_time_array)
test_time_array_40 = np.zeros_like(test_time_array)

test_time_array_02[:,:,0] = test_time_array[:,:,0]
test_time_array_02[:,:,1] = test_time_array[:,:,1] + t_shift*time_step/total_time

test_time_array_04[:,:,0] = test_time_array[:,:,0]
test_time_array_04[:,:,1] = test_time_array[:,:,1] + 2*t_shift*time_step/total_time

test_time_array_20[:,:,1] = test_time_array[:,:,1]
test_time_array_20[:,:,0] = test_time_array[:,:,0] + t_shift*time_step/total_time

test_time_array_40[:,:,1] = test_time_array[:,:,1]
test_time_array_40[:,:,0] = test_time_array[:,:,0] + 2*t_shift*time_step/total_time

# Concatenate
train_dec0_concatenate = np.stack((train_dec0[:, None, :, :],time_train_dec0[:, None, :, :]), axis = 1)
train_dec1_concatenate = np.stack((train_dec1[:, None, :, :],time_train_dec1[:, None, :, :]), axis = 1)

val_dec0_concatenate = np.stack((val_dec0,val_time_dec0[:, None, :, :]), axis = 1)
val_dec1_concatenate = np.stack((val_dec1,val_time_dec1[:, None, :, :]), axis = 1)

TEST_00_concatenated = np.stack((TEST_00, test_time_array[:, None, :, :]), axis = 1)
TEST_02_concatenated = np.stack((TEST_02, test_time_array_02[:, None, :, :]), axis = 1)
TEST_20_concatenated = np.stack((TEST_20, test_time_array_20[:, None, :, :]), axis = 1)
TEST_04_concatenated = np.stack((TEST_04, test_time_array_04[:, None, :, :]), axis = 1)
TEST_40_concatenated = np.stack((TEST_40, test_time_array_40[:, None, :, :]), axis = 1)
TEST = np.concatenate((TEST_02_concatenated, TEST_00_concatenated, TEST_20_concatenated, TEST_04_concatenated, TEST_40_concatenated), axis = 0)


# Create Dataset / DataLoaders
train_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec0).float(), torch.from_numpy(np.expand_dims(REF_train_dec0, axis = -1)).float())
train_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec1).float(), torch.from_numpy(np.expand_dims(REF_train_dec1, axis = -1)).float())

val_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(val_dec0).float(), torch.from_numpy(np.expand_dims(REF_val_dec0, axis = -1)).float())
val_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(val_dec1).float(), torch.from_numpy(np.expand_dims(REF_val_dec1, axis = -1)).float())

train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = 32, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = 32, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = len(val_dataset_dec0), shuffle = False)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = len(val_dataset_dec1), shuffle = False)


# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model_dec0 = ConvolutionalModel(int(window_low + window_high))
model_dec1 = ConvolutionalModel(int(window_low + window_high))

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr, weight_decay = 1e-5) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr, weight_decay = 1e-5) 

#Execute train loop
loss_dec0, test_dec0, val_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/Conv_model_dec0',  save = save) 
loss_dec1, test_dec1, val_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/Conv_model_dec1',  save = save)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V02 = TOF[:,:TEST_00.shape[0]] 
TOF_V00 = TOF[:,TEST_00.shape[0] : 2*TEST_00.shape[0]] 
TOF_V20 = TOF[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]] 
TOF_V04 = TOF[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]] 
TOF_V40 = TOF[:,4*TEST_00.shape[0]:] 
    
    
# Calculate centered position 'centroid'
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + t_shift*time_step))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - t_shift*time_step))
error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 2*t_shift*time_step))
error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 2*t_shift*time_step))

Error = np.concatenate((error_V02, error_V00, error_V20, error_V04, error_V40), axis = 1)   

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
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Train loss Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Train loss Detector 1')
plt.plot(np.log10(val_loss_dec0.astype('float32')), label = 'Val loss Detector 0')
plt.plot(np.log10(val_loss_dec1.astype('float32')), label = 'Val loss Detector 1')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plot_gaussian(TOF_V04[-1,:], centroid_V00[-1], range = 0.8, label = '-0.4 ns offset', nbins = nbins)
plot_gaussian(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V40[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.4 ns offset', nbins = nbins)

params_V04, errors_V04 = get_gaussian_params(TOF_V04[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V02, errors_V02 = get_gaussian_params(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V40, errors_V40 = get_gaussian_params(TOF_V40[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)

print("V40: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
print("V20: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
print("V04: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)')
plt.ylabel('Counts')
plt.show()

#import time
#
#time_test = np.tile(train_dec0[0,:,:], (1000000, 1,1))
#
#time_list_inference = []
## Start timer inference
#for i in range(10):
#    start_time_inference= time.time()
#    with torch.no_grad():
#        assert not torch.is_grad_enabled()
#        output_time_test = model_dec0(torch.tensor(M_time_test[:,:,0]).float().to(device))
#    end_time_inference = time.time()
#    elapsed_time_inference = end_time_inference - start_time_inference
#    time_list_inference.append(elapsed_time_inference)
#time_array_inference = np.array(time_list_inference)
#
#print('Elapsed time momentos:', np.mean(time_array_moments), np.std(time_array_moments))
#print('Elapsed time inference:', np.mean(time_array_inference), np.std(time_array_inference))