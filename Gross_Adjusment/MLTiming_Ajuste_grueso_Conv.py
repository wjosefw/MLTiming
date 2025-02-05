import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Import functions 
from functions import (move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, calculate_gaussian_center, 
                       plot_gaussian, get_gaussian_params, get_mean_pulse_from_set)
from Models import ConvolutionalModel,  count_parameters
from Train_loops import train_loop_convolutional

# Load data 
dir = '/home/josea/Pulsos15CM20250130/'

data0 = np.load(os.path.join(dir, 'Na22_norm_pos0.npz'))['data']
data1 = np.load(os.path.join(dir, 'Na22_norm_pos1.npz'))['data']
data2 = np.load(os.path.join(dir, 'Na22_norm_pos2.npz'))['data']
data3 = np.load(os.path.join(dir, 'Na22_norm_pos3.npz'))['data']
data4 = np.load(os.path.join(dir, 'Na22_norm_pos4.npz'))['data']
data5 = np.load(os.path.join(dir, 'Na22_norm_pos5.npz'))['data']
data6 = np.load(os.path.join(dir, 'Na22_norm_pos6.npz'))['data']

data_min_1 = np.load(os.path.join(dir, 'Na22_norm_pos_min_1.npz'))['data']
data_min_2 = np.load(os.path.join(dir, 'Na22_norm_pos_min_2.npz'))['data']
data_min_3 = np.load(os.path.join(dir, 'Na22_norm_pos_min_3.npz'))['data']
data_min_4 = np.load(os.path.join(dir, 'Na22_norm_pos_min_4.npz'))['data']
data_min_5 = np.load(os.path.join(dir, 'Na22_norm_pos_min_5.npz'))['data']
data_min_6 = np.load(os.path.join(dir, 'Na22_norm_pos_min_6.npz'))['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_time = 0.8                # Max delay to training pulses in ns
time_step = 0.2                 # Signal time step in ns
set_seed(42)                    # Fix seeds
nbins = 71                     # Num bins for all histograms
positions = 0.066*np.array([6, 5, 4, 3, 2, 1, 0.0, -1, -2, -3, -4, -5, -6])  # Expected time difference of each position
start_dec0 = 262
stop_dec0 = 275
start_dec1 = 262
stop_dec1 = 275
lr = 1e-5
epochs = 50
batch_size = 32
save = True

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = data0[:20000,:,:]
validation_data = data0[20000:21000,:,:]
test_data = np.concatenate((data0[-10000:,:,:], data1[-10000:,:,:],  data2[-10000:,:,:],  data3[-10000:,:,:],
                            data4[-10000:,:,:],  data5[-10000:,:,:],  data6[-10000:,:,:],
                            data_min_1[-10000:,:,:], data_min_2[-10000:,:,:], data_min_3[-10000:,:,:],
                            data_min_4[-10000:,:,:], data_min_5[-10000:,:,:], data_min_6[-10000:,:,:]), axis = 0)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

mean_pulse_dec0 = get_mean_pulse_from_set(train_data, channel = 0)
mean_pulse_dec1 = get_mean_pulse_from_set(train_data, channel = 1)

plt.plot(mean_pulse_dec0[start_dec0:stop_dec0])
plt.plot(mean_pulse_dec1[start_dec1:stop_dec1])
plt.show()

np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/reference_pulse_dec0.npz', data = mean_pulse_dec0)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/reference_pulse_dec1.npz', data = mean_pulse_dec1)

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

# Create Dataloaders
train_loader_dec0 = create_dataloaders(train_dec0, REF_train_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = create_dataloaders(train_dec1, REF_train_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0  = create_dataloaders(val_dec0, REF_val_dec0, batch_size = batch_size, shuffle = False)
val_loader_dec1  = create_dataloaders(val_dec1, REF_val_dec1, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

set_seed(42)
model_dec0 = ConvolutionalModel(int(stop_dec0-start_dec0))
set_seed(42)
model_dec1 = ConvolutionalModel(int(stop_dec1-start_dec1))

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr, weight_decay = 1e-5) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr, weight_decay = 1e-5) 

#Execute train loop
loss_dec0, test_dec0, val_loss_dec0, val_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/AG_model_dec0',  save = save) 
loss_dec1, test_dec1, val_loss_dec1, val_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/AG_model_dec1',  save = save)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)

TOF_0 = TOF[:,:10000] 
TOF_1 = TOF[:, 10000:20000 ] 
TOF_2 = TOF[:, 20000:30000] 
TOF_3 = TOF[:, 30000:40000] 
TOF_4 = TOF[:, 40000:50000] 
TOF_5 = TOF[:, 50000:60000] 
TOF_6 = TOF[:, 60000:70000] 
TOF_min_1 = TOF[:, 70000:80000] 
TOF_min_2 = TOF[:, 80000:90000] 
TOF_min_3 = TOF[:, 90000:100000] 
TOF_min_4 = TOF[:, 100000:110000] 
TOF_min_5 = TOF[:, 110000:120000] 
TOF_min_6 = TOF[:, 120000:] 


# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_0, nbins = nbins, limit = 1) 

error_min_6 = abs((TOF_min_6 - centroid_V00[:, np.newaxis] - positions[0]))
error_min_5 = abs((TOF_min_5 - centroid_V00[:, np.newaxis] - positions[1]))
error_min_4 = abs((TOF_min_4 - centroid_V00[:, np.newaxis] - positions[2]))
error_min_3 = abs((TOF_min_3 - centroid_V00[:, np.newaxis] - positions[3]))
error_min_2 = abs((TOF_min_2 - centroid_V00[:, np.newaxis] - positions[4]))
error_min_1 = abs((TOF_min_1 - centroid_V00[:, np.newaxis] - positions[5]))
error_0 = abs((TOF_0 - centroid_V00[:, np.newaxis] - positions[6]))
error_1 = abs((TOF_1 - centroid_V00[:, np.newaxis] - positions[7]))
error_2 = abs((TOF_2 - centroid_V00[:, np.newaxis] - positions[8]))
error_3 = abs((TOF_3 - centroid_V00[:, np.newaxis] - positions[9]))
error_4 = abs((TOF_1 - centroid_V00[:, np.newaxis] - positions[10]))
error_5 = abs((TOF_2 - centroid_V00[:, np.newaxis] - positions[11]))
error_6 = abs((TOF_3 - centroid_V00[:, np.newaxis] - positions[12]))



# Get MAE
Error = np.concatenate((error_0, error_1, error_2,  error_3,
                        error_4, error_5, error_6, 
                        error_min_1,  error_min_2, error_min_3,
                        error_min_4,  error_min_5, error_min_6), axis = 1)   
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
plt.hist(test_dec0[-1, :delays_test_dec0.shape[0]] - time_step*delays_test_dec0, bins = nbins, range = [-1, 3], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[-1, :delays_test_dec1.shape[0]] - time_step*delays_test_dec1, bins = nbins, range = [-1, 3], alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(133)
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Training loss Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Training loss Detector 1')
plt.plot(np.log10(val_loss_dec0.astype('float32')), label = 'Val loss Detector 0')
plt.plot(np.log10(val_loss_dec1.astype('float32')), label = 'Val loss Detector 1')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plot_gaussian(TOF_0[-1,:], centroid_V00[-1], range = 0.6, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_3[-1,:], centroid_V00[-1], range = 0.6, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_5[-1,:], centroid_V00[-1], range = 0.6, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_min_3[-1,:], centroid_V00[-1], range = 0.6, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_min_5[-1,:], centroid_V00[-1], range = 0.6, label = '-0.2 ns offset', nbins = nbins)

params_0, errors_0 = get_gaussian_params(TOF_0[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_1, errors_1 = get_gaussian_params(TOF_1[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_2, errors_2 = get_gaussian_params(TOF_2[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_3, errors_3 = get_gaussian_params(TOF_3[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_4, errors_4 = get_gaussian_params(TOF_4[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_5, errors_5 = get_gaussian_params(TOF_5[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_6, errors_6 = get_gaussian_params(TOF_6[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_min_1, errors_min_1 = get_gaussian_params(TOF_min_1[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_min_2, errors_min_2 = get_gaussian_params(TOF_min_2[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_min_3, errors_min_3 = get_gaussian_params(TOF_min_3[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_min_4, errors_min_4 = get_gaussian_params(TOF_min_4[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_min_5, errors_min_5 = get_gaussian_params(TOF_min_5[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_min_6, errors_min_6 = get_gaussian_params(TOF_min_6[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)

print("min 6: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_6[2], errors_min_6[2], params_min_6[3], errors_min_6[3]))
print("min 5: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_5[2], errors_min_5[2], params_min_5[3], errors_min_5[3]))
print("min 4: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_4[2], errors_min_4[2], params_min_4[3], errors_min_4[3]))
print("min 3: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_3[2], errors_min_3[2], params_min_3[3], errors_min_3[3]))
print("min 2: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_2[2], errors_min_2[2], params_min_2[3], errors_min_2[3]))
print("min 1: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_1[2], errors_min_1[2], params_min_1[3], errors_min_1[3]))
print("0: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_0[2], errors_0[2], params_0[3], errors_0[3]))
print("1: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_1[2], errors_1[2], params_1[3], errors_1[3]))
print("2: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_2[2], errors_2[2], params_2[3], errors_2[3]))
print("3: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_3[2], errors_3[2], params_3[3], errors_3[3]))
print("4: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_4[2], errors_4[2], params_4[3], errors_4[3]))
print("5: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_5[2], errors_5[2], params_5[3], errors_5[3]))
print("6: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_6[2], errors_6[2], params_6[3], errors_6[3]))


print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

# -------------------------------------------------------------------------
# -------------------- SAVE RESULTS OVER EPOCHS ---------------------------
# -------------------------------------------------------------------------


idx = 20
TOF_0 = TOF_0[idx:]
TOF_1 = TOF_1[idx:]
TOF_2 = TOF_2[idx:]
TOF_3 = TOF_3[idx:]
TOF_4 = TOF_4[idx:]
TOF_5 = TOF_5[idx:]
TOF_6 = TOF_6[idx:]
TOF_min_1 = TOF_min_1[idx:]
TOF_min_2 = TOF_min_2[idx:]
TOF_min_3 = TOF_min_3[idx:]
TOF_min_4 = TOF_min_4[idx:]
TOF_min_5 = TOF_min_5[idx:]
TOF_min_6 = TOF_min_6[idx:]

REF_val_dec0 = REF_val_dec0[idx:]
REF_val_dec1 = REF_val_dec1[idx:]
val_dec0 = val_dec0[idx:,:,:] 
val_dec1 = val_dec1[idx:,:,:]
MAE = MAE[idx:]

centroid_0 = calculate_gaussian_center(TOF_0, nbins = nbins) 
centroid_1 = calculate_gaussian_center(TOF_1 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 
centroid_2 = calculate_gaussian_center(TOF_2 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1)
centroid_3 = calculate_gaussian_center(TOF_3 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 
centroid_4 = calculate_gaussian_center(TOF_4 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 
centroid_5 = calculate_gaussian_center(TOF_5 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1)
centroid_6 = calculate_gaussian_center(TOF_6 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 
centroid_min_1 = calculate_gaussian_center(TOF_min_1 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 
centroid_min_2 = calculate_gaussian_center(TOF_min_2 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1)
centroid_min_3 = calculate_gaussian_center(TOF_min_3 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 
centroid_min_4 = calculate_gaussian_center(TOF_min_4 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 
centroid_min_5 = calculate_gaussian_center(TOF_min_5 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1)
centroid_min_6 = calculate_gaussian_center(TOF_min_6 - centroid_V00[:, np.newaxis], nbins = nbins, limit = 1) 


error_min_6_centroid = abs((TOF_min_6 - positions[0]))
error_min_5_centroid = abs((TOF_min_5 - positions[1]))
error_min_4_centroid = abs((TOF_min_4 - positions[2]))
error_min_3_centroid = abs((TOF_min_3 - positions[3]))
error_min_2_centroid = abs((TOF_min_2 - positions[4]))
error_min_1_centroid = abs((TOF_min_1 - positions[5]))

error_0_centroid = abs((TOF_0 - positions[6]))
error_1_centroid = abs((TOF_1 - positions[7]))
error_2_centroid = abs((TOF_2 - positions[8]))
error_3_centroid = abs((TOF_3 - positions[9]))
error_4_centroid = abs((TOF_4 - positions[10]))
error_5_centroid = abs((TOF_5 - positions[11]))
error_6_centroid = abs((TOF_6 - positions[12]))


avg_bias = np.mean(np.stack((error_0_centroid , error_1_centroid, error_2_centroid, error_3_centroid,
                             error_4_centroid, error_5_centroid, error_6_centroid,
                             error_min_1_centroid,  error_min_2_centroid, error_min_3_centroid,
                             error_min_4_centroid,  error_min_5_centroid, error_min_6_centroid), axis = -1), axis = 1)


# Plot MAE_singles vs MAE_coincidences
err_val_dec0 = abs(val_dec0[:,:,0] - val_dec0[:,:,1] - REF_val_dec0[np.newaxis,:])
err_val_dec1 = abs(val_dec1[:,:,0] - val_dec1[:,:,1] - REF_val_dec1[np.newaxis,:])
mean_err_val_dec0 = np.mean(err_val_dec0, axis = 1)
mean_err_val_dec1 = np.mean(err_val_dec1, axis = 1)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec0_Na22.npz', data = mean_err_val_dec0)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec1_Na22.npz', data = mean_err_val_dec1)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/MAE_Na22.npz', data = MAE)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/avg_bias.npz', data = avg_bias)

CTR = []
for i in range(TOF_0.shape[0]):
    params_0, errors_0 = get_gaussian_params(TOF_0[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_1, errors_1 = get_gaussian_params(TOF_1[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_2, errors_2 = get_gaussian_params(TOF_2[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_3, errors_3 = get_gaussian_params(TOF_3[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_4, errors_4 = get_gaussian_params(TOF_4[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_5, errors_5 = get_gaussian_params(TOF_5[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_6, errors_6 = get_gaussian_params(TOF_6[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_min_1, errors_min_1 = get_gaussian_params(TOF_min_1[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_min_2, errors_min_2 = get_gaussian_params(TOF_min_2[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_min_3, errors_min_3 = get_gaussian_params(TOF_min_3[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_min_4, errors_min_4 = get_gaussian_params(TOF_min_4[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_min_5, errors_min_5 = get_gaussian_params(TOF_min_5[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    params_min_6, errors_min_6 = get_gaussian_params(TOF_min_6[i,:], centroid_V00[i], range = 0.6, nbins = nbins)
    CTR.append(np.mean([params_0[3],  params_1[3],  params_2[3], params_3[3], 
                         params_4[3],  params_5[3], params_6[3],  
                        params_min_1[3],  params_min_2[3], params_min_3[3],
                        params_min_4[3],  params_min_5[3], params_min_6[3]]))
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/ctr.npz', data = np.array(CTR))

