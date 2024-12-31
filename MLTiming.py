import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import (momentos, create_and_delay_pulse_pair, create_position, 
                       set_seed, calculate_gaussian_center, normalize, 
                       normalize_given_params, plot_gaussian, get_gaussian_params,
                       continuous_delay)
from Models import MLP_Torch,  count_parameters
from Train_loops import train_loop_MLP, train_loop_KAN

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

delay_time = 1                        # Max delay to training pulses in ns
time_step = 0.2                       # Signal time step in ns
moments_order = int(sys.argv[1])      # Max order of moments used
set_seed(42)                          # Fix seeds
nbins = 71                            # Num bins for all histograms                   
t_shift = 1                           # Time steps to move for the new positions
normalization_method = 'standardization'
start = 47
stop = 74
lr = 1e-4
epochs = 500
batch_size = 32
Num_Neurons = 8
architecture = [moments_order, int(sys.argv[2]), 1, 1]    # KAN architecture
save = False


# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------

train_data = np.concatenate((train_data_55, train_data_28, train_data_82), axis = 0)
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

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

TEST = test_data


# Calculate moments 
M_Train_dec0 = momentos(train_dec0, order = moments_order) 
M_Train_dec1 = momentos(train_dec1, order = moments_order) 

M_Val_dec0 = momentos(val_dec0, order = moments_order) 
M_Val_dec1 = momentos(val_dec1, order = moments_order) 

MOMENTS_TEST = momentos(TEST, order = moments_order)

# Normalize moments
M_Train_dec0, params_dec0 =  normalize(M_Train_dec0, method = normalization_method)
M_Train_dec1, params_dec1 =  normalize(M_Train_dec1, method = normalization_method)

M_Val_dec0_channel0 =  normalize_given_params(M_Val_dec0, params_dec0, channel = 0, method = normalization_method)
M_Val_dec0_channel1 =  normalize_given_params(M_Val_dec0, params_dec0, channel = 1, method = normalization_method)
M_Val_dec0 = np.stack((M_Val_dec0_channel0, M_Val_dec0_channel1), axis = -1)

M_Val_dec1_channel0 =  normalize_given_params(M_Val_dec1, params_dec1, channel = 0, method = normalization_method)
M_Val_dec1_channel1 =  normalize_given_params(M_Val_dec1, params_dec1, channel = 1, method = normalization_method)
M_Val_dec1 = np.stack((M_Val_dec1_channel0, M_Val_dec1_channel1), axis = -1)

MOMENTS_TEST_norm_dec0 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
MOMENTS_TEST_norm_dec1 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)
MOMENTS_TEST = np.stack((MOMENTS_TEST_norm_dec0, MOMENTS_TEST_norm_dec1), axis = -1)


# Create Datasets/Dataloaders
train_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(M_Train_dec0).float(), torch.from_numpy(np.expand_dims(REF_train_dec0, axis = -1)).float())
train_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(M_Train_dec1).float(), torch.from_numpy(np.expand_dims(REF_train_dec1, axis = -1)).float())

val_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(M_Val_dec0).float(), torch.from_numpy(np.expand_dims(REF_val_dec0, axis = -1)).float())
val_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(M_Val_dec1).float(), torch.from_numpy(np.expand_dims(REF_val_dec1, axis = -1)).float())

train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = len(val_dataset_dec0), shuffle = False)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = len(val_dataset_dec1), shuffle = False)

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
#loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(MOMENTS_TEST[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec0', save = save) 
#loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(MOMENTS_TEST[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec1', save = save)
loss_dec0, val_loss_dec0, test_dec0 = train_loop_MLP(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(MOMENTS_TEST[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/MLP_model_dec0', save = save) 
loss_dec1, val_loss_dec1, test_dec1 = train_loop_MLP(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(MOMENTS_TEST[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/MLP_model_dec1', save = save)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------


# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V00 = TOF[:,:test_data_55.shape[0]] 
TOF_V02 = TOF[:, test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
TOF_V20 = TOF[:, test_data_55.shape[0]  + test_data_28.shape[0]:] 
    

# Calulate Test error
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 5) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + time_step*t_shift))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - time_step*t_shift))


#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1) 
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
#
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