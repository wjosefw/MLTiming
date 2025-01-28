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

delay_time = 0.8                # Max delay to training pulses in ns
time_step = 0.2               # Signal time step in ns
set_seed(42)                  # Fix seeds
nbins = 71                    # Num bins for all histograms
positions = [-0.2, 0.0, 0.2]  # Expected time difference of each position
start = 60
stop = 74
lr = 1e-5
epochs = 375
batch_size = 32
save = True

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = np.concatenate((train_data_55, validation_data_55, train_data_55),axis = 0)
validation_data = np.concatenate((validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

mean_pulse_dec0 = get_mean_pulse_from_set(train_data, channel = 0)
mean_pulse_dec1 = get_mean_pulse_from_set(train_data, channel = 1)

np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/reference_pulse_dec0.npz', data = mean_pulse_dec0)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/reference_pulse_dec1.npz', data = mean_pulse_dec1)

# Train/Validation/Test set
delays_dec0, moved_pulses_dec0 = move_to_reference(mean_pulse_dec0, train_data, start = start, stop = stop, max_delay = int(stop-start), channel = 0)
delays_dec1, moved_pulses_dec1 = move_to_reference(mean_pulse_dec1, train_data, start = start, stop = stop, max_delay = int(stop-start), channel = 1)

delays_val_dec0, moved_pulses_val_dec0 = move_to_reference(mean_pulse_dec0, validation_data, start = start, stop = stop, max_delay = int(stop-start), channel = 0)
delays_val_dec1, moved_pulses_val_dec1 = move_to_reference(mean_pulse_dec1, validation_data, start = start, stop = stop, max_delay = int(stop-start), channel = 1)

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start, stop = stop, max_delay = int(stop-start), channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start, stop = stop, max_delay = int(stop-start), channel = 1)

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
model_dec0 = ConvolutionalModel(int(stop-start))
set_seed(42)
model_dec1 = ConvolutionalModel(int(stop-start))

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

TOF_V00 = TOF[:,:test_data_55.shape[0]] 
TOF_V02 = TOF[:, test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
TOF_V20 = TOF[:, test_data_55.shape[0]  + test_data_28.shape[0]:] 

# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[0]))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[2]))

#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1)   
MAE = np.mean(Error, axis = 1)
print(MAE[-1])

#Plot
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
plot_gaussian(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)

params_V02, errors_V02 = get_gaussian_params(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)

print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

### Combine the two numbers
#num = f"{sys.argv[1]}{sys.argv[2]}"


# Your existing variables
#FWHM = np.array([params_V02[3], params_V00[3], params_V20[3]])  # ps
#FWHM_err = np.array([errors_V02[3],  errors_V00[3],  errors_V20[3]])        # ps
#centroid = np.array([params_V02[2], params_V00[2], params_V20[2]])  # ps
#centroid_err = np.array([errors_V02[2],  errors_V00[2],  errors_V20[2]])        # ps
#
## Multiply by 1000
#FWHM = FWHM * 1000
#FWHM_err = FWHM_err * 1000
#centroid = centroid * 1000
#centroid_err = centroid_err * 1000
#
#with open('results_FS.txt', 'a') as file:
#    file.write(f"FWHM_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM)}])  # ps\n")
#    file.write(f"FWHM_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM_err)}])  # ps\n")
#    file.write(f"centroid_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid)}])  # ps\n")
#    file.write(f"centroid_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid_err)}])  # ps\n")
#    file.write(f"MAE_{num} = {MAE[-1]:.7f}  # ps\n")  # Write MAE with one decima
#    file.write(f"mean_FWHM_{num} = np.mean(FWHM_{num})\n")
#    file.write(f"mean_FWHM_err_{num} = np.mean(FWHM_err_{num})\n")
#    file.write(f"mean_bias_{num} = np.mean(abs(centroid_{num} - positions))\n")
#    file.write("\n")  # Add a new line for better separation
#
#TOF_V00 = TOF_V00[10:]
#TOF_V02 = TOF_V02[10:]
#TOF_V20 = TOF_V20[10:]

centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins) 
centroid_V02 = calculate_gaussian_center(TOF_V02 - centroid_V00[:, np.newaxis], nbins = nbins) 
centroid_V20 = calculate_gaussian_center(TOF_V20 - centroid_V00[:, np.newaxis], nbins = nbins)

error_V20_centroid = abs(centroid_V20 - 0.2)
error_V02_centroid = abs(centroid_V02 + 0.2)

avg_bias = np.mean(np.stack((error_V20_centroid , error_V02_centroid), axis = -1), axis = 1)


#Plot MAE_singles vs MAE_coincidences
err_val_dec0 = abs(val_dec0[:,:,0] - val_dec0[:,:,1] - REF_val_dec0[np.newaxis,:])
err_val_dec1 = abs(val_dec1[:,:,0] - val_dec1[:,:,1] - REF_val_dec1[np.newaxis,:])
mean_err_val_dec0 = np.mean(err_val_dec0, axis = 1)
mean_err_val_dec1 = np.mean(err_val_dec1, axis = 1)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec0_Na22.npz', data = mean_err_val_dec0)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec1_Na22.npz', data = mean_err_val_dec1)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/MAE_Na22.npz', data = MAE)
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/avg_bias.npz', data = avg_bias)

CTR = []
for i in range(TOF_V00.shape[0]):
    params_V02, errors_V02 = get_gaussian_params(TOF_V02[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
    params_V00, errors_V00 = get_gaussian_params(TOF_V00[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
    params_V20, errors_V20 = get_gaussian_params(TOF_V20[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
    CTR.append(np.mean([params_V20[3],params_V00[3],params_V02[3]]))
np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/ctr.npz', data = np.array(CTR))

