import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import Hyperparameters and Paths
from config_Fixed_Window import (device, delay_time, time_step, nbins, 
                                 seed, epochs, lr, batch_size, save, 
                                 start, stop, BASE_DIR, MODEL_SAVE_DIR, 
                                 DATA_DIR)

print(device)
sys.path.append(str(BASE_DIR.parent))

# Import functions
from functions import (create_and_delay_pulse_pair, set_seed, Calculate_CFD,
                       create_dataloaders)
from Models import ConvolutionalModel,  count_parameters
from Train_loops import train_loop_convolutional, train_loop_convolutional_with_target

# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

channel = 0
epochs2 = 150
set_seed(seed)                    # Fix seeds
save_name = os.path.join(MODEL_SAVE_DIR, 'FW_model_dec' + str(channel))

# -------------------------------------------------------------------------
#---------------------------- LOAD DATA -----------------------------------
# -------------------------------------------------------------------------

train_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_train.npz'), mmap_mode = 'r')['data']
validation_data = np.load(os.path.join(DATA_DIR, 'Na22_norm_pos0_val.npz'), mmap_mode = 'r')['data']

train_data = train_data[:, start:stop, channel]
validation_data = validation_data[:, start:stop, channel] 

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de validacion: ', validation_data.shape[0])
set_seed(seed)                    # Fix seeds

# -------------------------------------------------------------------------
# ---------------------- GET CFD TIMESTAMPS -------------------------------
# -------------------------------------------------------------------------

# Get CFD timestamps
timestamps_train = Calculate_CFD(train_data, fraction = 0.05, shift = 6.8, time_step = time_step)
timestamps_val = Calculate_CFD(validation_data, fraction = 0.05, shift = 6.8, time_step = time_step)

# -------------------------------------------------------------------------
# ----------------- PREPARE DATA FOR FIRST LOOP ---------------------------
# -------------------------------------------------------------------------

# Create Dataloaders
train_loader = create_dataloaders(train_data, timestamps_train, batch_size = batch_size, shuffle = True)
val_loader = create_dataloaders(validation_data, timestamps_val, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model = ConvolutionalModel(int(stop-start))
print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr) 

# -------------------------------------------------------------------------
# -------------------- FIRST TRAINING LOOP --------------------------------
# -------------------------------------------------------------------------

# Execute first loop
loss_target, val_loss_target = train_loop_convolutional_with_target(model, optimizer, train_loader, val_loader, EPOCHS = epochs, name = save_name, save = save) 

# Plot training and validation losses
plt.plot(np.log10(loss_target.astype('float32')), label = 'Train loss Detector 0')
plt.plot(np.log10(val_loss_target.astype('float32')), label = 'Val loss Detector 0')
plt.ylabel('Loss value (log)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

val = np.squeeze(model(torch.tensor(validation_data[:,None,:]).to(device).float()).cpu().detach().numpy())
plt.plot(timestamps_val, val - timestamps_val, 'b.')
plt.xlabel('Validation timestamps')
plt.ylabel('Validation Errors')
plt.show()

plt.hist(timestamps_val, bins = 71, alpha = 0.5,  label = 'True timestamps')
plt.hist(val, bins = 71, alpha = 0.5, label = 'Predicted timestamps')
plt.legend()
plt.show()

# -------------------------------------------------------------------------
# ------------------ CREATE VIRTUAL COINCIDENCES --------------------------
# -------------------------------------------------------------------------

# Create virtual coincidences
train, REF_train = create_and_delay_pulse_pair(train_data, time_step, delay_time = delay_time)
val, REF_val = create_and_delay_pulse_pair(validation_data, time_step, delay_time = delay_time)

# Create Dataloaders
train_loader = create_dataloaders(train, REF_train, batch_size = batch_size, shuffle = True)
val_loader = create_dataloaders(val, REF_val, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------- USE VIRTUAL COINCIDENCES TO CHECK RESULTS OF FIRST LOOP ---------
# -------------------------------------------------------------------------

# Make a prediction of virtual coincidences
test0 = np.squeeze(model(torch.tensor(val[:,None,:,0]).to(device).float()).cpu().detach().numpy())
test1 = np.squeeze(model(torch.tensor(val[:,None,:,1]).to(device).float()).cpu().detach().numpy())

# Plot validation delays hists
plt.hist(test0 - test1, bins = nbins, range = [-2, 2], alpha = 0.5, label = 'Prediction')
plt.hist(REF_val, bins = nbins, range = [-2, 2], alpha = 0.5, label = 'Reference')
plt.legend()
plt.show()

# Get validation error
err_val = test0 - test1 - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

# Plot validation delay vs error
plt.plot(REF_val, err_val, 'b.', markersize = 1.5)
plt.xlabel('Validation target delay')
plt.ylabel('Validation prediction error')
plt.show()

# -------------------------------------------------------------------------
# -------------------- SECOND TRAINING LOOP -------------------------------
# -------------------------------------------------------------------------

# Execute train loop
loss, val_loss, test, val_inf = train_loop_convolutional(model, optimizer, train_loader, val_loader, torch.tensor(np.zeros_like(validation_data)).float(), EPOCHS = epochs2, name = save_name,  save = save) 

# -------------------------------------------------------------------------
# ------------------- SECOND TRAINING LOOP RESULTS ------------------------
# -------------------------------------------------------------------------

# Plot
plt.plot(np.log10(loss.astype('float32')), label = 'Train loss Detector ' + str(channel))
plt.plot(np.log10(val_loss.astype('float32')), label = 'Val loss Detector ' + str(channel))
plt.ylabel('Loss value (log)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Plot validation delays hists
plt.hist(val_inf[-1,:,0] - val_inf[-1,:,1], bins = nbins, alpha = 0.5, label = 'Predicted delays validation')
plt.hist(REF_val, bins = nbins, alpha = 0.5, label = 'Target delays validation')
plt.legend()
plt.show()

# Get validation error
err_val = (val_inf[-1,:,0] - val_inf[-1,:,1]) - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

# Plot target validation delay vs error
plt.plot(REF_val, err_val, 'b.', markersize = 1.5)
plt.xlabel('Validation target delay')
plt.ylabel('Validation prediction error')
plt.show()




