import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Import Hyperparameters and Paths
from config_Fixed_Window import (device, delay_time, time_step, nbins, 
                                 seed, epochs, lr, batch_size, save, 
                                 positions, start, stop, BASE_DIR,
                                 MODEL_SAVE_DIR)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions
from functions import (create_and_delay_pulse_pair, calculate_gaussian_center, 
                       plot_gaussian, get_gaussian_params, set_seed,
                       create_dataloaders)
from Models import ConvolutionalModel,  count_parameters
from Train_loops import train_loop_convolutional


# Load data 
dir = '/home/josea/PracticaTimingDigital'
data = np.load(os.path.join(dir, 'pulsoNa22:filt_norm.npz'))['data']
set_seed(seed)                 # Fix seeds

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = data[:10000,:,:]
validation_data = data[:1,:,:]
test_data = data[10000:,:,:]

train_data = train_data[:,start:stop,:]
validation_data = validation_data[:,start:stop,:] 
test_data = test_data[:,start:stop,:]

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

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

val_loader_dec0  = create_dataloaders(val_dec0, REF_val_dec0, batch_size = batch_size, shuffle = False)
val_loader_dec1  = create_dataloaders(val_dec1, REF_val_dec1, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

set_seed(seed)
model_dec0 = ConvolutionalModel(int(stop-start))
set_seed(seed)
model_dec1 = ConvolutionalModel(int(stop-start))

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr, weight_decay = 1e-5) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr, weight_decay = 1e-5) 

#Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(test_data[:,:,0]).float(), EPOCHS = epochs, name = 'Trained_Models/Fixed_Window_model_dec0',  save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(test_data[:,:,1]).float(), EPOCHS = epochs, name = 'Trained_Models/Fixed_Window_model_dec1',  save = save)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = test_dec0 - test_dec1
print(TOF.shape)
TOF_V00 = TOF

   
# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limit = 3) 
    
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))

# Get MAE
#Error = np.concatenate((error_V02, error_V20, error_V00),  axis = 1)  
#MAE = np.mean(Error, axis = 1)
MAE = np.mean(error_V00, axis = 1)
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

plot_gaussian(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)


params_V00, errors_V00 = get_gaussian_params(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)

print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[1], errors_V00[2], params_V00[2], errors_V00[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

#centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 
#centroid_V02 = calculate_gaussian_center(TOF_V02 - centroid_V00[:, np.newaxis], nbins = nbins, limits = 3) 
#centroid_V20 = calculate_gaussian_center(TOF_V20 - centroid_V00[:, np.newaxis], nbins = nbins, limits = 3)
#
#error_V20_centroid = abs(centroid_V20 - 0.2)
#error_V02_centroid = abs(centroid_V02 + 0.2)
#
#avg_bias = np.mean(np.stack((error_V20_centroid , error_V02_centroid), axis = -1), axis = 1)
#
#
#Plot MAE_singles vs MAE_coincidences
#err_val_dec0 = abs(val_dec0[:,:,0] - val_dec0[:,:,1] - REF_val_dec0[np.newaxis,:])
#err_val_dec1 = abs(val_dec1[:,:,0] - val_dec1[:,:,1] - REF_val_dec1[np.newaxis,:])
#mean_err_val_dec0 = np.mean(err_val_dec0, axis = 1)
#mean_err_val_dec1 = np.mean(err_val_dec1, axis = 1)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec0_Na22.npz', data = mean_err_val_dec0)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec1_Na22.npz', data = mean_err_val_dec1)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/MAE_Na22.npz', data = MAE)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/avg_bias.npz', data = avg_bias)


#CTR = []
#for i in range(epochs):
#    params_V02, errors_V02 = get_gaussian_params(TOF_V02[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
#    params_V00, errors_V00 = get_gaussian_params(TOF_V00[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
#    params_V20, errors_V20 = get_gaussian_params(TOF_V20[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
#    CTR.append(np.mean([params_V20[3],params_V00[3],params_V02[3]]))
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/ctr.npz', data = np.array(CTR))
#

#### Combine the two numbers
#num = f"{sys.argv[1]}{sys.argv[2]}"
#
## Your existing variables
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
#
## Open the file in append mode
#with open('results_FS.txt', 'a') as file:
#    file.write(f"FWHM_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM)}])  # ps\n")
#    file.write(f"FWHM_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM_err)}])  # ps\n")
#    file.write(f"centroid_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid)}])  # ps\n")
#    file.write(f"centroid_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid_err)}])  # ps\n")
#    file.write(f"MAE_{num} = {MAE[-1]:.7f}  # ps\n")  # Write MAE with one decima
#    file.write("\n")  # Add a new line for better separation