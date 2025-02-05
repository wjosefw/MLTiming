import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Import functions 
from functions import (momentos, move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, calculate_gaussian_center, 
                       normalize, normalize_given_params, plot_gaussian, 
                       get_gaussian_params, get_mean_pulse_from_set)
from Models import MLP_Torch,  count_parameters
from Train_loops import train_loop_KAN, train_loop_MLP

# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'

train_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_train.npz'))['data']
train_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_train.npz'))['data']
train_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_train.npz'))['data']

validation_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_val.npz'))['data']
validation_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_val.npz'))['data']
validation_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_val.npz'))['data']

#test_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_test.npz'))['data']
#test_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_test.npz'))['data']
#test_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_test.npz'))['data']

test_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_val_FS.npz'))['data']
test_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_val_FS.npz'))['data']
test_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_val_FS.npz'))['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

import sys
delay_time = 0.8                          # Max delay to training pulses in ns
time_step = 0.2                         # Signal time step in ns
moments_order = int(sys.argv[1])        # Max order of moments used
set_seed(42)                            # Fix seeds
nbins = 71                              # Num bins for all histograms
positions = [-0.2, 0.0, 0.2]            # Expected time difference of each position
normalization_method = 'min-max'
start_dec0 = 60
stop_dec0 = 74
start_dec1 = 61
stop_dec1 = 75
lr = 1e-4
epochs = 1000
Num_Neurons = int(sys.argv[2])
batch_size = 32  
architecture = [moments_order, 5, 1, 1]    # KAN architecture
save = True

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = train_data_55
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

mean_pulse_dec0 = get_mean_pulse_from_set(train_data, channel = 0)
mean_pulse_dec1 = get_mean_pulse_from_set(train_data, channel = 1)

#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/reference_pulse_dec0.npz', data = mean_pulse_dec0)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/reference_pulse_dec1.npz', data = mean_pulse_dec1)

# Train/Validation/Test set
delays_dec0, moved_pulses_dec0 = move_to_reference(mean_pulse_dec0, train_data, start = start_dec0, stop = stop_dec0, max_delay = int(stop_dec0-start_dec0), channel = 0)
delays_dec1, moved_pulses_dec1 = move_to_reference(mean_pulse_dec1, train_data, start = start_dec1, stop = stop_dec1, max_delay = int(stop_dec1-start_dec1), channel = 1)

delays_val_dec0, moved_pulses_val_dec0 = move_to_reference(mean_pulse_dec0, validation_data, start = start_dec0, stop = stop_dec0, max_delay = int(stop_dec0-start_dec0), channel = 0)
delays_val_dec1, moved_pulses_val_dec1 = move_to_reference(mean_pulse_dec1, validation_data, start = start_dec1, stop = stop_dec1, max_delay = int(stop_dec1-start_dec1), channel = 1)

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start_dec0, stop = stop_dec0, max_delay = int(stop_dec0-start_dec0), channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start_dec1, stop = stop_dec1, max_delay = int(stop_dec1-start_dec1), channel = 1)

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

# Create Dataloaders
train_loader_dec0 = create_dataloaders(M_Train_dec0, REF_train_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = create_dataloaders(M_Train_dec1, REF_train_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0  = create_dataloaders(M_Val_dec0, REF_val_dec0, batch_size = batch_size, shuffle = False)
val_loader_dec1  = create_dataloaders(M_Val_dec1, REF_val_dec1, batch_size = batch_size, shuffle = False)

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

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr)  

# Execute train loop
#loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(M_Test[:,:,0]).float(), EPOCHS = epochs, save = save) 
#loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(M_Test[:,:,1]).float(), EPOCHS = epochs, save = save)
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_MLP(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(M_Test[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/MLP_AG_model_dec0', save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_MLP(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(M_Test[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/MLP_AG_model_dec1', save = save)

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
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Detector 1')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.legend()
#plt.show()

  
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
#plt.show()


### Combine the two numbers
num = f"{sys.argv[1]}{sys.argv[2]}"
 
#Your existing variables
FWHM = np.array([params_V02[3], params_V00[3], params_V20[3]])  # ps
FWHM_err = np.array([errors_V02[3],  errors_V00[3],  errors_V20[3]])        # ps
centroid = np.array([params_V02[2], params_V00[2], params_V20[2]])  # ps
centroid_err = np.array([errors_V02[2],  errors_V00[2],  errors_V20[2]])        # ps

# Multiply by 1000
FWHM = FWHM * 1000
FWHM_err = FWHM_err * 1000
centroid = centroid * 1000
centroid_err = centroid_err * 1000

with open('results_FS.txt', 'a') as file:
    file.write(f"FWHM_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM)}])  # ps\n")
    file.write(f"FWHM_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM_err)}])  # ps\n")
    file.write(f"centroid_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid)}])  # ps\n")
    file.write(f"centroid_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid_err)}])  # ps\n")
    file.write(f"MAE_{num} = {MAE[-1]:.7f}  # ps\n")  # Write MAE with one decima
    file.write(f"mean_FWHM_{num} = np.mean(FWHM_{num})\n")
    file.write(f"mean_FWHM_err_{num} = np.mean(FWHM_err_{num})\n")
    file.write(f"mean_bias_{num} = np.mean(abs(centroid_{num} - positions))\n")
    file.write("\n")  # Add a new line for better separation

    

#centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins) 
#centroid_V02 = calculate_gaussian_center(TOF_V02 - centroid_V00[:, np.newaxis], nbins = nbins) 
#centroid_V20 = calculate_gaussian_center(TOF_V20 - centroid_V00[:, np.newaxis], nbins = nbins)
#
#error_V20_centroid = abs(centroid_V20 - 0.2)
#error_V02_centroid = abs(centroid_V02 + 0.2)
#
#avg_bias = np.mean(np.stack((error_V20_centroid , error_V02_centroid), axis = -1), axis = 1)
#
#
##Plot MAE_singles vs MAE_coincidences
#err_val_dec0 = abs(val_dec0[:,:,0] - val_dec0[:,:,1] - REF_val_dec0[np.newaxis,:])
#err_val_dec1 = abs(val_dec1[:,:,0] - val_dec1[:,:,1] - REF_val_dec1[np.newaxis,:])
#mean_err_val_dec0 = np.mean(err_val_dec0, axis = 1)
#mean_err_val_dec1 = np.mean(err_val_dec1, axis = 1)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec0_Na22.npz', data = mean_err_val_dec0)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec1_Na22.npz', data = mean_err_val_dec1)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/MAE_Na22.npz', data = MAE)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/avg_bias.npz', data = avg_bias)
#
#CTR = []
#for i in range(TOF_V00.shape[0]):
#    params_V02, errors_V02 = get_gaussian_params(TOF_V02[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
#    params_V00, errors_V00 = get_gaussian_params(TOF_V00[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
#    params_V20, errors_V20 = get_gaussian_params(TOF_V20[i,:], centroid_V00[i], range = 0.8, nbins = nbins)
#    CTR.append(np.mean([params_V20[3],params_V00[3],params_V02[3]]))
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/ctr.npz', data = np.array(CTR))
