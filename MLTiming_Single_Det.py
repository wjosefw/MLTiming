import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import Hyperparameters and Paths
from config import (
    device, seed, batch_size, epochs, learning_rate, Num_Neurons, before, after, 
    delay_time, nbins, DATA_DIR, MODEL_SAVE_DIR, REF_PULSE_SAVE_DIR, BASE_DIR, model_type,
    threshold, FIGURES_DIR  
)

print(device)
sys.path.append(str(BASE_DIR.parent))


# Import functions 
from functions import (move_to_reference, create_and_delay_pulse_pair, 
                       set_seed, create_dataloaders, get_mean_pulse_from_set,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel,  count_parameters, MLP_Torch
from Dataset import Datos_LAB_GFN
from Train_loops import train_loop

# Load data 
dataset = Datos_LAB_GFN(data_dir = DATA_DIR)

train_data = dataset.load_train_data()
validation_data = dataset.load_val_data()

time_step, positions, Theoretical_TOF = dataset.load_params() # Load data parameters

print('Number of training cases: ', train_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

channel = 0
set_seed(seed)                    # Fix seeds
save_name = f'{model_type}_AG_model_dec{str(channel)}'

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

# Calculate mean pulse
mean_pulse = get_mean_pulse_from_set(train_data, channel = channel)

np.savez_compressed(os.path.join(REF_PULSE_SAVE_DIR, 'reference_pulse_dec' + str(channel) + '.npz'), data = mean_pulse)

# Get start and stop
crossing = calculate_slope_y_intercept(mean_pulse, time_step, threshold = threshold)

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

if model_type == 'CNN':
    model = ConvolutionalModel(int(stop - start))
elif model_type == 'MLP':
    model = MLP_Torch(NM = int(stop- start), NN = Num_Neurons, STD_INIT = 0.5)
else:
    raise ValueError(f"Unsupported model_type: {model_type}. This routine is for 'MLP' or 'CNN' models only.")

print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) 

# Execute train loop
val = train_loop(model, optimizer, train_loader, val_loader, EPOCHS = epochs, model_name = save_name, model_dir = MODEL_SAVE_DIR, figure_dir = FIGURES_DIR, model_type = model_type) 

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

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








