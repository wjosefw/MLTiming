import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import Hyperparameters and Paths
from config_Gross_Adjustment import (
    device, seed, batch_size, epochs, learning_rate, Num_Neurons, before, after, save, 
    time_step, delay_time, nbins, DATA_DIR, MODEL_SAVE_DIR, REF_PULSE_SAVE_DIR, BASE_DIR
)

print(device)
sys.path.append(str(BASE_DIR.parent))


# Import functions 
from functions import (move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, get_mean_pulse_from_set,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel,  count_parameters, MLP_Torch
from Train_loops import train_loop_convolutional_single_det, train_loop_MLP_Single_Det

# Load data 
train_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_train.npz'), mmap_mode = 'r')['data']
validation_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_val.npz'), mmap_mode = 'r')['data']

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de validacion: ', validation_data.shape[0])

# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

channel = 0
set_seed(seed)                    # Fix seeds
save_name = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec' + str(channel))

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

# Calculate mean pulse
mean_pulse = get_mean_pulse_from_set(train_data, channel = channel)

np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, 'reference_pulse_dec' + str(channel) + '.npz'), data = mean_pulse)

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

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) 

# Execute train loop
#loss, val_loss, val = train_loop_convolutional_single_det(model, optimizer, train_loader, val_loader, EPOCHS = epochs, name = save_name,  save = save) 
loss, val_loss, val = train_loop_MLP_Single_Det(model, optimizer, train_loader, val_loader, EPOCHS = epochs, name = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_AG_model_dec' + str(channel)), save = save)

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








