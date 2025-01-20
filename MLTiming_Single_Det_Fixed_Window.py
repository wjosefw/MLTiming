import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN


from functions import (set_seed, create_and_delay_pulse_pair,
                       Calculate_CFD, create_dataloaders)
from Models import  count_parameters, ConvolutionalModel
from Train_loops import train_loop_convolutional, train_loop_convolutional_with_target


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
set_seed(42)                               # Fix seeds
nbins = 71                                 # Num bins for all histograms
start = 47 
stop = 74 
time_step = 0.2                            # Signal time step in ns
epochs = 500                               # Number of epochs for training (first loop)
epochs2 = 500                              # Number of epochs for training (second loop) 
lr = 1e-3                                  # Model learning rate
batch_size = 32                            # batch size used for training
save = False                               # Save models or not
save_name = 'predictions/Convolutional/Conv_model_dec' + str(channel)
fraction_cfd = 0.064                       # Fraction for CFD 
shift = 6.8                                # Shift for CFD 

train_data = np.concatenate((train_data_55, train_data_28, train_data_82), axis = 0)
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

# -------------------------------------------------------------------------
# ---------------------- GET CFD TIMESTAMPS -------------------------------
# -------------------------------------------------------------------------

# Get CFD timestamps
timestamps_train = Calculate_CFD(train_data[:,:200, channel], fraction = fraction_cfd, shift = shift, time_step = time_step)
timestamps_val = Calculate_CFD(validation_data[:,:200, channel], fraction = fraction_cfd, shift = shift, time_step = time_step)
timestamps_test = Calculate_CFD(test_data[:,:200, channel], fraction = fraction_cfd, shift = shift, time_step = time_step)

# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------

train_data = train_data[:,start:stop,channel]
validation_data = validation_data[:,start:stop,channel] 
test_data = test_data[:,start:stop,channel]

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de validacion: ', validation_data.shape[0])

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

# Execute train loop
loss, test, val_loss, val_inf = train_loop_convolutional(model, optimizer, train_loader, val_loader, torch.tensor(test_data).float(), EPOCHS = epochs2, name = save_name,  save = save) 

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

plt.hist(test[-1,:], bins = nbins, alpha = 0.5, label = 'Detector ' + str(channel));
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()

#plot validation delays hists
plt.hist(val_inf[-1,:,0] - val_inf[-1,:,1], bins = nbins, alpha = 0.5, label = 'Predicted delays validation')
plt.hist(REF_val, bins = nbins, alpha = 0.5, label = 'Target delays validation')
plt.legend()
plt.show()

# Get validation error
err_val = (val_inf[-1,:,0] - val_inf[-1,:,1]) - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

#Plot target validation delay vs error
plt.plot(REF_val, err_val, 'b.', markersize = 1.5)
plt.xlabel('Validation target delay')
plt.ylabel('Validation prediction error')
plt.show()




