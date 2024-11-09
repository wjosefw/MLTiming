import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


from functions import (momentos_threshold, create_position, set_seed, calculate_gaussian_center, 
                       normalize, normalize_given_params, plot_gaussian, get_gaussian_params, 
                       continuous_delay, extract_signal_along_time, create_and_delay_pulse_pair_along_time)
from Train_loops import train_loop_KAN


# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'

#train_data = np.load(os.path.join(dir,'Na22_train.npz'))['data']
#val_data = np.load(os.path.join(dir, 'Na22_val.npz'))['data']
#test_data = np.load(os.path.join(dir, 'Na22_test_val.npz'))['data']
data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA.npz'))['data']
data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA.npz'))['data']
data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA.npz'))['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_time = 0.2                             # Max delay to training pulses in ns
moments_order = 5                          # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 71                                # Num bins for all histograms
t_shift = 1                                # Time steps to move for the new positions
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
epochs = 500                              # Number of epochs for training
lr = 1e-3                                  # Model learning rate
batch_size = 256                            # batch size used for training
total_time = time_step*data_55.shape[1] - time_step  #time_step*train_data.shape[1] - time_step 
save = False                               # Save models or not
architecture = [moments_order, 5, 1, 1]   
fraction = 0.1                             # Fraction to trigger the pulse cropping   
window_low = 14                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger


# -------------------------------------------------------------------------
#----------------------- ALIGN PULSES -------------------------------------
# -------------------------------------------------------------------------

#align_time = 0.56
#new_train = continuous_delay(train_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)
#new_val = continuous_delay(val_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)
#new_test = continuous_delay(test_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)

align_time = 0.1
new_data_82 = continuous_delay(data_82, time_step = time_step, delay_time = align_time, channel_to_fix = 1, channel_to_move = 0)
new_data_55 = continuous_delay(data_55, time_step = time_step, delay_time = align_time, channel_to_fix = 1, channel_to_move = 0)
new_data_28 = continuous_delay(data_28, time_step = time_step, delay_time = align_time, channel_to_fix = 1, channel_to_move = 0)

new_train = np.concatenate((new_data_55[:6000,:,:], new_data_28[:6000,:,:], new_data_82[:6000,:,:]), axis = 0)
new_val = np.concatenate((new_data_55[6000:7000,:,:], new_data_28[6000:7000,:,:], new_data_82[6000:7000,:,:]), axis = 0)
new_test = np.concatenate((new_data_55[6000:,:,:], new_data_28[6000:,:,:], new_data_82[6000:,:,:]), axis = 0)


#print('Número de casos de entrenamiento: ', train_data.shape[0])
#print('Número de casos de test: ', test_data.shape[0])
# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train_array, train_time_array, a, b = extract_signal_along_time(new_train, time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)
val_array, val_time_array, _ , _    = extract_signal_along_time(new_val, time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)
test_array, test_time_array, _ , _  = extract_signal_along_time(new_test, time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)
print(a,b)
train_dec0, REF_train_dec0, time_train_dec0 = create_and_delay_pulse_pair_along_time(train_array[:,:,0], train_time_array[:,:,0], time_step, total_time, delay_time = delay_time)
train_dec1, REF_train_dec1, time_train_dec1 = create_and_delay_pulse_pair_along_time(train_array[:,:,1], train_time_array[:,:,1], time_step, total_time, delay_time = delay_time)

val_dec0, REF_val_dec0, val_time_dec0 = create_and_delay_pulse_pair_along_time(val_array[:,:,0], val_time_array[:,:,0], time_step, total_time, delay_time = delay_time)
val_dec1, REF_val_dec1, val_time_dec1 = create_and_delay_pulse_pair_along_time(val_array[:,:,1], val_time_array[:,:,1], time_step, total_time, delay_time = delay_time)

#TEST_00 = test_array 
TEST_00 = test_array
#TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift)
#TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift)
#TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift))
#TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift))

for i in range(test_array.shape[0]):
    plt.plot(test_array[i,:,0],'b-')
    plt.plot(test_array[i,:,1],'r-')
plt.show()
# Calculate moments
M_Train_dec0 = momentos_threshold(train_dec0, time_train_dec0, order = moments_order) 
M_Train_dec1 = momentos_threshold(train_dec1, time_train_dec1, order = moments_order) 

M_Val_dec0 = momentos_threshold(val_dec0, val_time_dec0, order = moments_order) 
M_Val_dec1 = momentos_threshold(val_dec1, val_time_dec1, order = moments_order)

MOMENTS_TEST_00 = momentos_threshold(TEST_00, test_time_array, order = moments_order)

#MOMENTS_TEST_02_0 = momentos_threshold(TEST_02[:,:,0], test_time_array[:,:,0], order = moments_order)
#MOMENTS_TEST_02_1 = momentos_threshold(TEST_02[:,:,1], test_time_array[:,:,1] + t_shift*time_step/total_time, order = moments_order)
#MOMENTS_TEST_02 = np.stack((MOMENTS_TEST_02_0, MOMENTS_TEST_02_1), axis = -1)
#
#MOMENTS_TEST_04_0 = momentos_threshold(TEST_04[:,:,0], test_time_array[:,:,0], order = moments_order)
#MOMENTS_TEST_04_1 = momentos_threshold(TEST_04[:,:,1], test_time_array[:,:,1] + 2*t_shift*time_step/total_time, order = moments_order)
#MOMENTS_TEST_04 = np.stack((MOMENTS_TEST_04_0, MOMENTS_TEST_04_1), axis = -1)
#                           
#MOMENTS_TEST_20_1 = momentos_threshold(TEST_20[:,:,1], test_time_array[:,:,1], order = moments_order)
#MOMENTS_TEST_20_0 = momentos_threshold(TEST_20[:,:,0], test_time_array[:,:,0] + t_shift*time_step/total_time, order = moments_order)
#MOMENTS_TEST_20 = np.stack((MOMENTS_TEST_20_0, MOMENTS_TEST_20_1), axis = -1)
#
#MOMENTS_TEST_40_1 = momentos_threshold(TEST_40[:,:,1], test_time_array[:,:,1], order = moments_order)
#MOMENTS_TEST_40_0 = momentos_threshold(TEST_40[:,:,0], test_time_array[:,:,0] + 2*t_shift*time_step/total_time, order = moments_order)
#MOMENTS_TEST_40 = np.stack((MOMENTS_TEST_40_0, MOMENTS_TEST_40_1), axis = -1)

#MOMENTS_TEST = np.concatenate((MOMENTS_TEST_02, MOMENTS_TEST_00, MOMENTS_TEST_20, MOMENTS_TEST_04, MOMENTS_TEST_40), axis = 0)
MOMENTS_TEST = MOMENTS_TEST_00

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

plt.hist(M_Train_dec0[:,0,0], range = [-5, 5], bins = 71, alpha = 0.5);
plt.hist(MOMENTS_TEST[:,0,0], range = [-5, 5], bins = 71, alpha = 0.5);
plt.show()

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

# Print information 
print("Normalization parameters detector 0:", params_dec0)
print("Normalization parameters detector 1:", params_dec1)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

# Initialize model
model_dec0 = KAN(architecture)
model_dec1 = KAN(architecture)

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr)  

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0, val_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(MOMENTS_TEST[:,:,0]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec0', save = save) 
loss_dec1, val_loss_dec1, test_dec1, val_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(MOMENTS_TEST[:,:,1]).float(), EPOCHS = epochs, name = 'KAN_models/model_dec1', save = save)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = test_dec0 - test_dec1

#TOF_V02 = TOF[:,:TEST_00.shape[0]] 
#TOF_V00 = TOF[:,TEST_00.shape[0] : 2*TEST_00.shape[0]] 
#TOF_V20 = TOF[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]] 
#TOF_V04 = TOF[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]] 
#TOF_V40 = TOF[:,4*TEST_00.shape[0]:] 

TOF_V00 = TOF[:,:new_data_55[6000:,:,:].shape[0]] 
TOF_V02 = TOF[:, new_data_55[6000:,:,:].shape[0] : new_data_55[6000:,:,:].shape[0] + new_data_28[6000:,:,:].shape[0]] 
TOF_V20 = TOF[:, new_data_55[6000:,:,:].shape[0]  + new_data_28[6000:,:,:].shape[0]:] 
        
    
# Calculate centered position 'centroid'
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + t_shift*time_step))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - t_shift*time_step))
#error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 2*t_shift*time_step))
#error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 2*t_shift*time_step))

Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1)
#Error = np.concatenate((error_V02, error_V00, error_V20, error_V04, error_V40), axis = 1)   

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
plt.hist(test_dec0[-1,:], bins = nbins, alpha = 0.5, range = [-0.2, 1.5], label = 'Detector 0');
plt.hist(test_dec1[-1,:], bins = nbins, alpha = 0.5, range = [-0.2, 1.5], label = 'Detector 1');
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
#plot_gaussian(TOF_V04[-1,:], centroid_V00[-1], range = 0.8, label = '-0.4 ns offset', nbins = nbins)
plot_gaussian(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
#plot_gaussian(TOF_V40[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.4 ns offset', nbins = nbins)

#params_V04, errors_V04 = get_gaussian_params(TOF_V04[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V02, errors_V02 = get_gaussian_params(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
#params_V40, errors_V40 = get_gaussian_params(TOF_V40[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)

#print("V40: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
print("V20: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
#print("V04: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)')
plt.ylabel('Counts')
plt.show()
