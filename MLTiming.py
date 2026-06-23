import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt


# Import Hyperparameters and Paths
from Config import (
    device, seed, batch_size, epochs, learning_rate, Num_Neurons, before, after, FIGURES_DIR,
    moments_order, delay_time, nbins, threshold, normalization_method, MODEL_SAVE_DIR,
    architecture, model_type, model_name, time_step, train_data_path, validation_data_path, channel
)

print(device)

# Import functions
from functions import (momentos, create_and_delay_pulse_pair, set_seed,
                       create_dataloaders, normalize, normalize_given_params,
                       extract_signal_window_by_fraction, select_channel)
from Models import MLP_Torch,  ConvolutionalModel, count_parameters
from Train_loops import train_loop
from efficient_kan.src.efficient_kan import KAN

# -------------------------------------------------------------------------
#---------------------------- LOAD DATA -----------------------------------
# -------------------------------------------------------------------------

# Accepts either single-detector data (N, M) or paired coincidence data (N, M, 2),
# in which case `channel` (set in config.py) picks which detector to train on.
train_data = select_channel(np.load(train_data_path), channel = channel)
validation_data = select_channel(np.load(validation_data_path), channel = channel)

print('Number of training cases: ', train_data.shape[0])
print('Number of validation cases: ', validation_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

# Delays
moved_pulses_train, delays_train = extract_signal_window_by_fraction(train_data, time_step, fraction = threshold, window_low = before, window_high = after)
moved_pulses_val, delays_val  = extract_signal_window_by_fraction(validation_data, time_step, fraction = threshold, window_low = before, window_high = after)

train, REF_train = create_and_delay_pulse_pair(moved_pulses_train, time_step, delay_time = delay_time)
val, REF_val = create_and_delay_pulse_pair(moved_pulses_val, time_step, delay_time = delay_time)

params = None  # normalization params, only computed for moment-based models (KAN/MLP)

if model_type in ['KAN','MLP']:

    # Calculate moments
    M_Train = momentos(train, order = moments_order)
    M_Val = momentos(val, order = moments_order)

    # Normalize moments
    M_Train, params =  normalize(M_Train, method = normalization_method)

    M_Val_channel0 =  normalize_given_params(M_Val, params, channel = 0, method = normalization_method)
    M_Val_channel1 =  normalize_given_params(M_Val, params, channel = 1, method = normalization_method)
    M_Val = np.stack((M_Val_channel0, M_Val_channel1), axis = -1)

    # Create Dataloaders
    train_loader = create_dataloaders(M_Train, REF_train, batch_size = batch_size, shuffle = True)
    val_loader  = create_dataloaders(M_Val, REF_val, batch_size = M_Val.shape[0], shuffle = False)

    # Print information
    print("Normalization parameters:", params)

elif model_type in ['CNN','MLPWAVE']:

    # Create Dataloaders
    train_loader = create_dataloaders(train, REF_train, batch_size = batch_size, shuffle = True)
    val_loader  = create_dataloaders(val, REF_val, batch_size = val.shape[0], shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

if model_type == 'KAN':
    model = KAN(architecture)

elif model_type == 'MLP':
    model = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)

elif model_type == 'CNN':
    model = ConvolutionalModel(int(before + after))

elif model_type == 'MLPWAVE':
    model = MLP_Torch(NM = int(before + after), NN = Num_Neurons, STD_INIT = 0.5)

else:
    raise ValueError(f"Unsupported model_type: {model_type}. This routine is for 'KAN', 'MLP', 'MLPWAVE' and 'CNN' models only.")
                  
print(f"Total number of parameters: {count_parameters(model)}")
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# Metadata needed to rebuild this model and its preprocessing pipeline at inference time
metadata = {
    'model_type': model_type,
    'architecture': architecture,
    'num_neurons': Num_Neurons,
    'moments_order': moments_order,
    'time_step_ns': time_step,
    'crossing_threshold': threshold,
    'before_samples': before,
    'after_samples': after,
    'normalization_method': normalization_method,
    'normalization_params_min': params[0].tolist() if params is not None else None,
    'normalization_params_max': params[1].tolist() if params is not None else None,
}

# Execute train loop
val = train_loop(model, optimizer, train_loader, val_loader, EPOCHS = epochs, model_name = model_name, model_dir = MODEL_SAVE_DIR, figure_dir = FIGURES_DIR, model_type = model_type, metadata = metadata)

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
