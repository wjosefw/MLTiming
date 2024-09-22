import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN
#from faster_kan.fastkan.fastkan import FastKAN
#from kan import *
from functions import (momentos, create_and_delay_pulse_pair, create_position, 
                       set_seed, calculate_gaussian_center_sigma, normalize, 
                       normalize_given_params, interpolate_pulses, plot_gaussian, 
                       get_gaussian_params)
from functions_KAN import  count_parameters, train_loop_KAN
from Models import train_loop_MLP, MLP_Torch

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

delay_steps = 30        # Max number of steps to delay pulses
moments_order = int(sys.argv[1])       # Max order of moments used
set_seed(42)            # Fix seeds
nbins = 91              # Num bins for all histograms                   
t_shift = 8             # Time steps to move for the new positions
normalization_method = 'standardization'
EXTRASAMPLING = 8
start_idx = 50
stop_idx = 74
start = start_idx*EXTRASAMPLING 
stop = stop_idx*EXTRASAMPLING 
lr = 1e-3
epochs = 500
Num_Neurons = 64

# -------------------------------------------------------------------------
#----------------------- INTERPOLATE PULSES -------------------------------
# -------------------------------------------------------------------------

new_train, new_time_step =  interpolate_pulses(train_data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)
new_val, new_time_step =  interpolate_pulses(val_data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)
new_test, new_time_step =  interpolate_pulses(test_data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)

# Align the pulses 
align_steps = 20

new_train[:,:,1] = np.roll(new_train[:,:,1], align_steps)
new_val[:,:,1] = np.roll(new_val[:,:,1], align_steps)
new_test[:,:,1] = np.roll(new_test[:,:,1], align_steps)

new_train[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)
new_val[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)
new_test[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)

print('New number of time points: %.d' % (new_train.shape[1]))
print('New time step: %.4f' % (new_time_step))

# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------

train_data = new_train[:,start:stop,:] 
validation_data = new_val[:,start:stop,:] 
test_data = new_val[:,start:stop,:]
print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])


# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(train_data[:,:,0], new_time_step, delay_steps = delay_steps, NOISE = False)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(train_data[:,:,1], new_time_step, delay_steps = delay_steps, NOISE = False)

val_dec0, REF_val_dec0 = create_and_delay_pulse_pair(validation_data[:,:,0], new_time_step, delay_steps = delay_steps, NOISE = False)
val_dec1, REF_val_dec1 = create_and_delay_pulse_pair(validation_data[:,:,1], new_time_step, delay_steps = delay_steps, NOISE = False)

TEST_00 = test_data 
TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift, NOISE = False)
TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift, NOISE = False)
TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift), NOISE = False)
TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift), NOISE = False)
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

train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = 32, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = 32, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = 32, shuffle = False)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = 32, shuffle = False)

# Print information 
print("NM dec0 =", M_Train_dec0.shape[1])
print("NM dec1 =", M_Train_dec1.shape[1])
print("Normalization parameters detector 0:", params_dec0)
print("Normalization parameters detector 1:", params_dec1)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

NM = M_Train_dec0.shape[1]
architecture = [NM, int(sys.argv[2]), 1, 1]    

model_dec0 = KAN(architecture)
model_dec1 = KAN(architecture)
#model_dec0 = MLP_Torch(NM = NM, NN = Num_Neurons, STD_INIT = 0.5)
#model_dec1 = MLP_Torch(NM = NM, NN = Num_Neurons, STD_INIT = 0.5)
         
print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr)  

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(MOMENTS_TEST[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec0', save = False) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(MOMENTS_TEST[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec1', save = False)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF_V02 = test_dec0[:,:TEST_00.shape[0]] - test_dec1[:,:TEST_00.shape[0]]
TOF_V00 = test_dec0[:,TEST_00.shape[0] : 2*TEST_00.shape[0]] - test_dec1[:,TEST_00.shape[0] : 2*TEST_00.shape[0]]
TOF_V20 = test_dec0[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]] - test_dec1[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]]
TOF_V04 = test_dec0[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]] - test_dec1[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]]
TOF_V40 = test_dec0[:,4*TEST_00.shape[0]:] - test_dec1[:,4*TEST_00.shape[0]:]
    

# Calulate Test error
centroid_V00, sigmaN_V00 = calculate_gaussian_center_sigma(TOF_V00, np.zeros((TOF_V00.shape[0])), nbins = nbins) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + 0.2))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - 0.2))
error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 0.4))
error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 0.4))

#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40), axis = 1)   
Error_No_V00 = np.concatenate((error_V02, error_V20, error_V04, error_V40), axis = 1) 
MAE = np.mean(Error, axis = 1)
MAE_No_V00 = np.mean(Error_No_V00, axis = 1)

idx_min_MAE = np.where(MAE_No_V00 == np.min(MAE_No_V00))[0][0]
print(idx_min_MAE, MAE[idx_min_MAE])


# Plot MAE_singles vs MAE_coincidences
#err_val_dec0 = abs(val_dec0[:,:,0] - val_dec0[:,:,1] - REF_val_dec0[np.newaxis,:])
#err_val_dec1 = abs(val_dec1[:,:,0] - val_dec1[:,:,1] - REF_val_dec1[np.newaxis,:])
#mean_err_val_dec0 = np.mean(err_val_dec0, axis = 1)
#mean_err_val_dec1 = np.mean(err_val_dec1, axis = 1)


# Plot
plt.figure(figsize = (20,5))
plt.subplot(131)
plt.plot(np.log10(MAE.astype('float64')), label = 'MAE')
plt.title('Results in coincidence')
plt.xlabel('Epochs')
plt.ylabel('Log10')
plt.legend()

plt.subplot(132)
plt.hist(test_dec0[idx_min_MAE,:], bins = nbins, range = [-1, 5], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[idx_min_MAE,:], bins = nbins, range = [-1, 5], alpha = 0.5, label = 'Detector 1');
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
#plt.show()



# Histogram and gaussian fit 
#plot_gaussian(TOF_V04[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = '-0.4 ns offset', nbins = nbins)
#plot_gaussian(TOF_V02[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
#plot_gaussian(TOF_V00[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
#plot_gaussian(TOF_V20[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
#plot_gaussian(TOF_V40[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.4 ns offset', nbins = nbins)


#params_V04, errors_V04 = get_gaussian_params(TOF_V04[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
#params_V02, errors_V02 = get_gaussian_params(TOF_V02[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
#params_V00, errors_V00 = get_gaussian_params(TOF_V00[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
#params_V20, errors_V20 = get_gaussian_params(TOF_V20[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
#params_V40, errors_V40 = get_gaussian_params(TOF_V40[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)


#print("V40: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
#print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
#print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
#print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
#print("V04: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

#print('')
#plt.legend()
#plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
#plt.ylabel('Counts', fontsize = 14)
#plt.show()


idx_min_MAE = -1
plot_gaussian(TOF_V04[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = '-0.4 ns offset', nbins = nbins)
plot_gaussian(TOF_V02[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V40[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.4 ns offset', nbins = nbins)


params_V04, errors_V04 = get_gaussian_params(TOF_V04[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
params_V02, errors_V02 = get_gaussian_params(TOF_V02[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)
params_V40, errors_V40 = get_gaussian_params(TOF_V40[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, nbins = nbins)


print("V40: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
print("V04: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
#plt.show()

# Combine the two numbers
num = f"{sys.argv[1]}{sys.argv[2]}"

# Your existing variables
FWHM = np.array([params_V04[3], params_V02[3], params_V00[3], params_V20[3], params_V40[3]])  # ps
FWHM_err = np.array([errors_V04[3],  errors_V02[3],  errors_V00[3],  errors_V20[3],  errors_V40[3]])        # ps
centroid = np.array([params_V04[2], params_V02[2], params_V00[2], params_V20[2], params_V40[2]])  # ps
centroid_err = np.array([errors_V04[2],  errors_V02[2],  errors_V00[2],  errors_V20[2],  errors_V40[2]])        # ps

# Multiply by 1000
FWHM = FWHM * 1000
FWHM_err = FWHM_err * 1000
centroid = centroid * 1000
centroid_err = centroid_err * 1000


# Open the file in append mode
with open('results_FS.txt', 'a') as file:
    file.write(f"FWHM_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM)}])  # ps\n")
    file.write(f"FWHM_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM_err)}])  # ps\n")
    file.write(f"centroid_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid)}])  # ps\n")
    file.write(f"centroid_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid_err)}])  # ps\n")
    file.write(f"MAE_{num} = {MAE[idx_min_MAE]:.7f}  # ps\n")  # Write MAE with one decima
    file.write("\n")  # Add a new line for better separation