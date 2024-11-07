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
from Models import train_loop_MLP, MLP_Torch,  count_parameters, train_loop_KAN

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
train_data = np.load(os.path.join(dir,'Na22_train.npz'))['data']
val_data = np.load(os.path.join(dir, 'Na22_val.npz'))['data']
test_data = np.load(os.path.join(dir, 'Na22_test_val.npz'))['data']


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
lr = 1e-3
epochs = 500
Num_Neurons = 16
architecture = [moments_order, int(sys.argv[2]), 1, 1]    # KAN architecture
save = False

# -------------------------------------------------------------------------
#----------------------- ALIGN PULSES -------------------------------------
# -------------------------------------------------------------------------

align_time = 0.5
new_train = continuous_delay(train_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)
new_val = continuous_delay(val_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)
new_test = continuous_delay(test_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)

# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------


#train_data = new_train[:,start:stop,:]  #189:213
#validation_data = new_val[:,start:stop,:] 
#test_data = new_test[:,start:stop,:]

train_data = np.concatenate((new_test[:,start:stop,:],new_train[:3000,start:stop,:]),axis = 0) 
validation_data = new_val[:,start:stop,:] 
test_data = new_train[3000:,start:stop,:]

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(train_data[:,:,0], time_step, delay_time = delay_time)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(train_data[:,:,1], time_step, delay_time = delay_time)

val_dec0, REF_val_dec0 = create_and_delay_pulse_pair(validation_data[:,:,0], time_step, delay_time = delay_time)
val_dec1, REF_val_dec1 = create_and_delay_pulse_pair(validation_data[:,:,1], time_step, delay_time = delay_time)


TEST_00 = test_data
TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift)
TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift)
TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift))
TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift))
TEST = np.concatenate((TEST_02, TEST_00, TEST_20, TEST_04, TEST_40), axis = 0)


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

train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = 128, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = 128, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = len(val_dataset_dec0), shuffle = False)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = len(val_dataset_dec1), shuffle = False)

# Print information 
print("Normalization parameters detector 0:", params_dec0)
print("Normalization parameters detector 1:", params_dec1)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model_dec0 = KAN(architecture)
model_dec1 = KAN(architecture)
#model_dec0 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
#model_dec1 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
         
print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr)  

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(MOMENTS_TEST[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec0', save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(MOMENTS_TEST[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec1', save = save)
#loss_dec0, val_loss_dec0, test_dec0 = train_loop_MLP(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(MOMENTS_TEST[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec0', save = save) 
#loss_dec1, val_loss_dec1, test_dec1 = train_loop_MLP(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(MOMENTS_TEST[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec1', save = save)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V02 = TOF[:,:TEST_00.shape[0]] 
TOF_V00 = TOF[:,TEST_00.shape[0] :2*TEST_00.shape[0]] 
TOF_V20 = TOF[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]] 
TOF_V04 = TOF[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]] 
TOF_V40 = TOF[:,4*TEST_00.shape[0]:] 
    

# Calulate Test error
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + time_step*t_shift))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - time_step*t_shift))
error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 2*time_step*t_shift))
error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 2*time_step*t_shift))

#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40), axis = 1)   
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


print("V40: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
print("V04: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

### Combine the two numbers
#num = f"{sys.argv[1]}{sys.argv[2]}"
#
## Your existing variables
#FWHM = np.array([params_V04[3], params_V02[3], params_V00[3], params_V20[3], params_V40[3]])  # ps
#FWHM_err = np.array([errors_V04[3],  errors_V02[3],  errors_V00[3],  errors_V20[3],  errors_V40[3]])        # ps
#centroid = np.array([params_V04[2], params_V02[2], params_V00[2], params_V20[2], params_V40[2]])  # ps
#centroid_err = np.array([errors_V04[2],  errors_V02[2],  errors_V00[2],  errors_V20[2],  errors_V40[2]])        # ps
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