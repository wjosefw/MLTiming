import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN


from functions import (momentos, create_and_delay_pulse_pair, set_seed, 
                       normalize, normalize_given_params, continuous_delay, 
                       plot_gaussian, get_gaussian_params)
from functions_KAN import  count_parameters, train_loop_KAN
from Models import train_loop_MLP, MLP_Torch

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'

train_data = np.load(os.path.join(dir,'Na22_train.npz'))['data']
val_data = np.load(os.path.join(dir, 'Na22_val.npz'))['data']

# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_time = 1          # Max delay to training pulses in ns
time_step = 0.2         # Signal time step in ns
moments_order = 5       # Max order of moments used
set_seed(42)            # Fix seeds
nbins = 91              # Num bins for all histograms                   
normalization_method = 'standardization'
start = 50
stop = 74
architecture = [moments_order, 5, 1, 1]  # KAN architecture
lr = 1e-3
epochs = 50
Num_Neurons = 64

# -------------------------------------------------------------------------
#----------------------- ALIGN PULSES -------------------------------------
# -------------------------------------------------------------------------

align_time = 0.6
new_train = continuous_delay(train_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)
new_val = continuous_delay(val_data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)

# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------

train_data = new_train[:,start:stop,:] 
validation_data = new_val[:,start:stop,:] 

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de validacion: ', validation_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train, REF_train = create_and_delay_pulse_pair(train_data[:,:,0], time_step, delay_steps = delay_time)
val, REF_val = create_and_delay_pulse_pair(validation_data[:,:,0],time_step, delay_steps = delay_time)

# Calculate moments 
M_Train = momentos(train, order = moments_order) 
M_Val = momentos(val, order = moments_order) 

# Normalize moments
M_Train, params=  normalize(M_Train, method = normalization_method)

M_Val_channel0 =  normalize_given_params(M_Val, params, channel = 0, method = normalization_method)
M_Val_channel1 =  normalize_given_params(M_Val, params, channel = 1, method = normalization_method)
M_Val = np.stack((M_Val_channel0, M_Val_channel1), axis = -1)

# Create Datasets/Dataloaders
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(M_Train).float(), torch.from_numpy(np.expand_dims(REF_train, axis = -1)).float())
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(M_Val).float(), torch.from_numpy(np.expand_dims(REF_val, axis = -1)).float())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle = False)

# Print information
print("Normalization parameters:", params)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model = KAN(architecture)
#model = MLP_Torch(NM = NM, NN = Num_Neurons, STD_INIT = 0.5)
print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr) 
  
# Execute train loop
loss, val_loss, test, val = train_loop_KAN(model, optimizer, train_loader, val_loader, torch.tensor(np.zeros_like(M_Train[:,:,0])).float(), EPOCHS = epochs, name = 'KAN_models/model', save = False) 
#loss, val_loss, test = train_loop_MLP(model, optimizer, train_loader, val_loader, torch.tensor(np.zeros_like(M_Train[:,:,0])).float(), EPOCHS = epochs, name = 'KAN_models/model', save = False) 

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Plot
plt.plot(np.log10(loss.astype('float32')), label = 'Train loss Detector 0')
plt.plot(np.log10(val_loss.astype('float32')), label = 'Val loss Detector 0')
plt.ylabel('Loss value (log)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Get validation error
err_val = val[-1,:,0] - val[-1,:,1] - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

# Plot Histogram and gaussian fit 
plot_gaussian(err_val, 0.0, range = 0.05, label = 'Validation errors', nbins = nbins)
params, errors = get_gaussian_params(err_val, 0.0, range = 0.05, nbins = nbins)
print("CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params[2], errors[2], params[3], errors[3]))

plt.legend()
plt.xlabel('$\epsilon$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()



