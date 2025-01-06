import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN


from functions import (extract_signal_along_time_singles, momentos_threshold, set_seed, 
                       normalize, normalize_given_params, calculate_slope_y_intercept,
                       create_and_delay_pulse_pair_along_time, create_dataloaders)
from Models import  count_parameters
from Train_loops import train_loop_KAN, train_loop_KAN_with_target

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
# ----------------------- IMPORTANT DEFINITIONS ---------------------------
# -------------------------------------------------------------------------

channel = 0                                # Channel to train
delay_time = 1                             # Max delay to training pulses in ns
moments_order = 7                          # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 71                                 # Num bins for all histograms
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
epochs = 500                                # Number of epochs for training (first loop)
epochs2 = 150                              # Number of epochs for training (second loop)
lr = 1e-3                                  # Model learning rate
batch_size = 32                            # batch size used for training 
save = True                                # Save models or not
save_name = 'KAN_models/model_dec' + str(channel)
architecture = [moments_order, 3, 1, 1]   
fraction = 0.1                             # Fraction to trigger the pulse cropping   
window_low = 14                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger


# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = np.concatenate((train_data_55, train_data_28, train_data_82), axis = 0)
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# ---------------- TRAINING TO REPRODUCE LED ------------------------------
# -------------------------------------------------------------------------

# Get time stamps
timestamps_train_list = []
for i in range(train_data.shape[0]):
  timestamp_train = calculate_slope_y_intercept(train_data[i,:,channel], time_step, threshold = fraction)
  timestamps_train_list.append(timestamp_train)


timestamps_val_list = []
for i in range(validation_data.shape[0]):
  timestamp_val = calculate_slope_y_intercept(validation_data[i,:,channel], time_step, threshold = fraction)
  timestamps_val_list.append(timestamp_val)


timestamps_test_list = []
for i in range(test_data.shape[0]):
  timestamp_test = calculate_slope_y_intercept(test_data[i,:,channel], time_step, threshold = fraction)
  timestamps_test_list.append(timestamp_test)

timestamps_train = np.array(timestamps_train_list)
timestamps_val = np.array(timestamps_val_list)
timestamps_test = np.array(timestamps_test_list)


# Extract pulses
train_array, train_time_array = extract_signal_along_time_singles(train_data[:, :100, channel], time_step, fraction = fraction, window_low = window_low, window_high = window_high)
val_array, val_time_array = extract_signal_along_time_singles(validation_data[:,:100,channel], time_step, fraction = fraction, window_low = window_low, window_high = window_high)
test_array, test_time_array = extract_signal_along_time_singles(test_data[:,:100,channel], time_step, fraction = fraction, window_low = window_low, window_high = window_high)


# Calculate moments
M_Train = momentos_threshold(train_array, train_time_array, order = moments_order) 
M_Val = momentos_threshold(val_array, val_time_array, order = moments_order) 
M_Test = momentos_threshold(test_array, test_time_array, order = moments_order)


# Normalize moments 
M_Train, n_params =  normalize(M_Train, method = normalization_method)
M_Val =  normalize_given_params(M_Val[:,:,None], n_params, channel = 0, method = normalization_method)
M_Test = normalize_given_params(M_Test[:,:,None], n_params, channel = 0, method = normalization_method)

print("Normalization parameters:", n_params)

# Create Dataloaders
train_loader = create_dataloaders(M_Train, timestamps_train, batch_size = batch_size, shuffle = True)
val_loader = create_dataloaders(M_Val, timestamps_val, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model = KAN(architecture)
print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr) 

# -------------------------------------------------------------------------
# -------------------- FIRST TRAINING LOOP -------------------------------
# -------------------------------------------------------------------------

# Execute first loop
loss_target, val_loss_target = train_loop_KAN_with_target(model, optimizer, train_loader, val_loader, EPOCHS = epochs, name = save_name, save = save) 

# Plot training and validation losses
plt.plot(np.log10(loss_target.astype('float32')), label = 'Train loss Detector 0')
plt.plot(np.log10(val_loss_target.astype('float32')), label = 'Val loss Detector 0')
plt.ylabel('Loss value (log)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# -------------------------------------------------------------------------
# ------------------ CREATE VIRTUAL COINCIDENCES --------------------------
# -------------------------------------------------------------------------

# virtual coincidences
train, REF_train, time_train = create_and_delay_pulse_pair_along_time(train_array, train_time_array, delay_time = delay_time)
val, REF_val, val_time = create_and_delay_pulse_pair_along_time(val_array, val_time_array, delay_time = delay_time)

# Calculate virtual coincidence moments
M_Train = momentos_threshold(train, time_train, order = moments_order) 
M_Val = momentos_threshold(val, val_time, order = moments_order) 

# Normalize virtual coincidence moments 
M_Train_channel0 =  normalize_given_params(M_Train, n_params, channel = 0, method = normalization_method)
M_Train_channel1 =  normalize_given_params(M_Train, n_params, channel = 1, method = normalization_method)
M_Train = np.stack((M_Train_channel0, M_Train_channel1), axis = -1)

M_Val_channel0 =  normalize_given_params(M_Val, n_params, channel = 0, method = normalization_method)
M_Val_channel1 =  normalize_given_params(M_Val, n_params, channel = 1, method = normalization_method)
M_Val = np.stack((M_Val_channel0, M_Val_channel1), axis = -1)

# Create virtual coincidences Dataloaders
train_loader = create_dataloaders(M_Train, REF_train, batch_size = batch_size, shuffle = True)
val_loader = create_dataloaders(M_Val, REF_val, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------- USE VIRTUAL COINCIDENCES TO CHECK RESULTS OF FIRST LOOP ---------
# -------------------------------------------------------------------------

test0 = np.squeeze(model(torch.tensor(M_Val[:,:,0]).to(device).float()).cpu().detach().numpy())
test1 = np.squeeze(model(torch.tensor(M_Val[:,:,1]).to(device).float()).cpu().detach().numpy())

#plot validation delays hists
plt.hist(test0 - test1, bins = nbins, range = [-2, 2], alpha = 0.5, label = 'Prediction')
plt.hist(REF_val, bins = nbins, range = [-2, 2], alpha = 0.5, label = 'Reference')
plt.legend()
plt.show()

# Get validation error
err_val = test0 - test1 - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

#Plot validation delay vs error
plt.plot(REF_val, err_val, 'b.', markersize = 1.5)
plt.xlabel('Validation target delay')
plt.ylabel('Validation prediction error')
plt.show()

# -------------------------------------------------------------------------
# -------------------- SECOND TRAINING LOOP -------------------------------
# -------------------------------------------------------------------------

# Execute second train loop using the previously created virtual coincidences
loss, val_loss, test, val = train_loop_KAN(model, optimizer, train_loader, val_loader, torch.tensor(M_Test).float(), EPOCHS = epochs2, name = save_name, save = save) 


# -------------------------------------------------------------------------
# ------------------- SECOND TRAINING LOOP RESULTS ------------------------
# -------------------------------------------------------------------------

# Plot
plt.plot(np.log10(loss.astype('float32')), label = 'Train loss Detector 0')
plt.plot(np.log10(val_loss.astype('float32')), label = 'Val loss Detector 0')
plt.ylabel('Loss value (log)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.hist(test[-1,:], bins = nbins, alpha = 0.5, label = 'Detector 0');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Get validation error
err_val = val[-1,:,0] - val[-1,:,1] - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

#plot validation delays hists
plt.hist(val[-1,:,0] - val[-1,:,1], range = [-2,2], bins = nbins, alpha = 0.5)
plt.hist(REF_val, bins = nbins, range = [-2,2], alpha = 0.5)
plt.show()

#Plot validation delay vs error
plt.plot(REF_val, err_val, 'b.', markersize = 1.5)
plt.ylabel('Validation Error')
plt.xlabel('Validation Target Delay')
plt.show()

