import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Import functions 
from functions import (move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, get_mean_pulse_from_set,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel,  count_parameters, MLP_Torch
from Train_loops import train_loop_convolutional_single_det, train_loop_MLP_Single_Det

# Load data 
dir = '/home/josea/Pulsos15CM20250130_version2/'

data0_train = np.load(os.path.join(dir, 'Na22_norm_pos0_train.npz'), mmap_mode='r')['data']
data0_val = np.load(os.path.join(dir, 'Na22_norm_pos0_val.npz'), mmap_mode='r')['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_time = 1                  # Max delay to training pulses in ns
time_step = 0.2                 # Signal time step in ns
set_seed(42)                    # Fix seeds
nbins = 51                      # Num bins for all histograms
lr = 1e-5
epochs = 500
batch_size = 32
before  = 8                     # Points to take before threshold
after = 5                       # Points to take after threshold
channel = 0
Num_Neurons = 16
save = True
save_name = '../Trained_Models/AG_model_dec' + str(channel)

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = data0_train
validation_data = data0_val

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de validacion: ', validation_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

# Calculate mean pulse
mean_pulse = get_mean_pulse_from_set(train_data, channel = channel)

np.savez_compressed('../predictions/reference_pulse_dec' + str(channel) + '.npz', data = mean_pulse)

# Get start and stop
crossing = calculate_slope_y_intercept(mean_pulse, time_step, threshold = 0.1)

start = int(crossing/time_step) - before
stop = int(crossing/time_step) + after

# Delays
delays, moved_pulses = move_to_reference(mean_pulse, train_data, start = start, stop = stop, channel = channel)
delays_val, moved_pulses_val = move_to_reference(mean_pulse, validation_data, start = start, stop = stop, channel = channel)

# Train/Validation/Test set
train, REF_train = create_and_delay_pulse_pair(moved_pulses, time_step, delay_time = delay_time)
val, REF_val = create_and_delay_pulse_pair(moved_pulses_val, time_step, delay_time = delay_time)

# Create Dataloaders
train_loader = create_dataloaders(train, REF_train, batch_size = batch_size, shuffle = True)
val_loader  = create_dataloaders(val, REF_val, batch_size = val.shape[0], shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

set_seed(42)
#model = ConvolutionalModel(int(stop - start))
model = MLP_Torch(NM = int(stop- start), NN = Num_Neurons, STD_INIT = 0.5)

print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-5) 

# Execute train loop
#loss, val_loss, val = train_loop_convolutional_single_det(model, optimizer, train_loader, val_loader, EPOCHS = epochs, name = save_name,  save = save) 
loss, val_loss, val = train_loop_MLP_Single_Det(model, optimizer, train_loader, val_loader, EPOCHS = epochs, name = '../Trained_Models/MLPWAVE_AG_model_dec' + str(channel), save = save)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------


# Plot
plt.plot(np.log10(loss.astype('float32')), label = 'Train loss Detector ' + str(channel))
plt.plot(np.log10(val_loss.astype('float32')), label = 'Val loss Detector ' + str(channel))
plt.ylabel('Loss value (log)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#plot validation delays hists
plt.hist(val[-1,:,0] - val[-1,:,1], bins = nbins, alpha = 0.5, label = 'Predicted delays validation')
plt.hist(REF_val, bins = nbins, alpha = 0.5, label = 'Target delays validation')
plt.legend()
plt.show()

# Get validation error
err_val = (val[-1,:,0] - val[-1,:,1]) - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

#Plot target validation delay vs error
plt.plot(REF_val, err_val, 'b.', markersize = 1.5)
plt.xlabel('Validation target delay')
plt.ylabel('Validation prediction error')
plt.show()








