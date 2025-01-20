import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import functions
from functions import (calculate_gaussian_center, plot_gaussian, get_gaussian_params, 
                       set_seed, create_dataloaders, calculate_slope_y_intercept)
from Models import ConvolutionalModel, count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvolutionalModel(nn.Module):
    def __init__(self, N_points, num_outputs=1):
        super(ConvolutionalModel, self).__init__()
        
        # --- same conv definitions as before ---
        self.conv1 = self.conv_block1(1, 32, (2,5)) 
        self.conv2 = self.conv_block1(32, 32, (1,3)) 
        self.conv3 = self.conv_block1(32, 64, (1,3))
        
        # Use actual final dimension:
        self.flatten_size = 64 * (N_points - 8)
        
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_outputs)

    def conv_block1(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1),
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_loop_convolutional(model, optimizer, train_loader, val_loader, EPOCHS = 75, name = 'model', save = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = torch.nn.L1Loss()  # Define MAE loss

    loss_list = []
    val_loss_list = []

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Make predictions for this batch 
            outputs = model(inputs[:,None,:,:])

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Calculate average loss per epoch
        avg_loss_epoch = running_loss / len(train_loader)
        loss_list.append(avg_loss_epoch)

        print(f'EPOCH {epoch + 1}: LOSS train {avg_loss_epoch}')

        # Calculate predictions on test_tensor
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                val = model(val_data[:,None,:,:])
                val_loss += loss_fn(val, val_labels).item()

            val_loss_list.append(val_loss / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

    if save:
        torch.save(model.state_dict(), name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype='object')
    val_loss = np.array(val_loss_list, dtype='object')

    return loss_array, val_loss

def continuous_delay(vector, delay_time, time_step = 0.2, channel_to_fix = 0, channel_to_move = 0):
    
    new_vector = np.zeros_like(vector)
    for i in range(vector.shape[0]):
        
        res = delay_time[i] % time_step  # Fractional part of the delay
        idel = int(delay_time[i] / time_step) 
        
        if delay_time[i] >= 0:
    
            for j in range(vector.shape[1] - 1, 0, -1):
                    slope = (vector[i, j, channel_to_move] - vector[i, j - 1, channel_to_move]) / time_step
                    new_vector[i, j, channel_to_move] =  vector[i, j, channel_to_move] - slope * res 
            new_vector[i,0, channel_to_move] = vector[i,0, channel_to_move]
            new_vector[i,:,channel_to_move] = np.roll(new_vector[i,:,channel_to_move], idel)
            new_vector[i,:idel,channel_to_move] = 0
            


        if delay_time[i] < 0:
            for j in range(vector.shape[1] - 1):
                    slope = (vector[i, j + 1, channel_to_move] - vector[i, j, channel_to_move]) / time_step
                    new_vector[i, j, channel_to_move] =  vector[i, j, channel_to_move] + slope * res 
            new_vector[i,-1, channel_to_move] = vector[i,-1, channel_to_move]
            if idel <= -1:
                new_vector[i,:,channel_to_move] = np.roll(new_vector[i,:,channel_to_move], idel)
                new_vector[i,idel:,channel_to_move] = vector[i, idel:, channel_to_move]
            else:
                pass
    new_vector[:,:,channel_to_fix] = vector[:,:,channel_to_fix]
    
    return new_vector

def continuous_delay2(vector, time_step = 0.2, delay_time = 1, channel_to_fix = 0, channel_to_move = 0):
    
    res = delay_time % time_step  # Fractional part of the delay
    idel = int(delay_time / time_step) 
    
    new_vector = np.zeros_like(vector)
    if delay_time >= 0:
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1] - 1, 0, -1):
                    slope = (vector[i, j, channel_to_move] - vector[i, j - 1, channel_to_move]) / time_step
                    new_vector[i, j, channel_to_move] =  vector[i, j, channel_to_move] - slope * res 
            new_vector[i,0, channel_to_move] = vector[i,0, channel_to_move]
            new_vector[i,:,channel_to_move] = np.roll(new_vector[i,:,channel_to_move], idel)
            new_vector[i,:idel,channel_to_move] = 0
        new_vector[:,:,channel_to_fix] = vector[:,:,channel_to_fix]


    if delay_time < 0:
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1] - 1):
                    slope = (vector[i, j + 1, channel_to_move] - vector[i, j, channel_to_move]) / time_step
                    new_vector[i, j, channel_to_move] =  vector[i, j, channel_to_move] + slope * res 
            new_vector[i,-1, channel_to_move] = vector[i,-1, channel_to_move]
            if idel <= -1:
                new_vector[i,:,channel_to_move] = np.roll(new_vector[i,:,channel_to_move], idel)
                new_vector[i,idel:,channel_to_move] = vector[i, idel:, channel_to_move]
            else:
                pass
        new_vector[:,:,channel_to_fix] = vector[:,:,channel_to_fix]
    return new_vector

def extract_signal_along_time(vector, time_step, fraction = 0.2, window_low = 140, window_high = 10):

    new_vector = np.zeros((vector.shape[0], int(window_high + window_low), 2))
    time_vector = np.zeros((vector.shape[0], int(window_high + window_low), 2))
    t = np.arange(0, time_step*vector.shape[1], time_step)

    a = 0
    b = 0    
    for i in range(vector.shape[0]):
        # Find indices where the signal in each channel exceeds the fraction threshold
        indices_channel0 = np.where(vector[i,:, 0] >= fraction)[0]
        indices_channel1 = np.where(vector[i,:, 1] >= fraction)[0]
        
        if indices_channel0[0] < indices_channel1[0]:
            index = indices_channel0[0]
            a += 1
        if indices_channel1[0] < indices_channel0[0]:
            index = indices_channel1[0]
            b += 1 
        elif indices_channel0[0] == indices_channel1[0]:
            index = indices_channel0[0]
             
        # Calculate the low and high indices to extraction
        index_low = index - window_low
        index_high = index + window_high

        # Extract cropped waveform and put into new vector
        new_vector[i,:, 0] =  vector[i,index_low:index_high, 0]
        new_vector[i,:, 1] =  vector[i,index_low:index_high, 1]

        time_vector[i,:,0] = t[index_low:index_high]
        time_vector[i,:,1] = t[index_low:index_high]

    return new_vector, time_vector, a, b    


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

time_step = 0.2     # Signal time step in ns
nbins = 71          # Num bins for all histograms                          
start = 60 
stop =  74 
set_seed(42)        # Fix seeds
epochs = 200
lr = 1e-4
batch_size = 32  
name = 'predictions/Convolutional/model_dec0'
save = False
threshold = 0.1
window_low = 14
window_high = 100

train_data = train_data_55
validation_data = validation_data_55
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

train_data = continuous_delay2(train_data, time_step = time_step, delay_time = 0.11, channel_to_fix = 1, channel_to_move = 0)
validation_data = continuous_delay2(validation_data, time_step = time_step, delay_time = 0.11, channel_to_fix = 1, channel_to_move = 0)
test_data = continuous_delay2(test_data, time_step = time_step, delay_time = 0.11, channel_to_fix = 1, channel_to_move = 0)

# -------------------------------------------------------------------------
#------------------------------ GET LED -----------------------------------
# -------------------------------------------------------------------------

timestamps_RLED_dec0_Train_list = []
timestamps_RLED_dec0_Val_list = []
timestamps_RLED_dec0_Test_list = []

timestamps_RLED_dec1_Train_list = []
timestamps_RLED_dec1_Val_list = []
timestamps_RLED_dec1_Test_list = []


for i in range(train_data.shape[0]):
  timestamp_dec0 = calculate_slope_y_intercept(train_data[i,:,0], time_step, threshold = threshold)
  timestamp_dec1 = calculate_slope_y_intercept(train_data[i,:,1], time_step, threshold = threshold)
  timestamps_RLED_dec0_Train_list.append(timestamp_dec0)
  timestamps_RLED_dec1_Train_list.append(timestamp_dec1)

for i in range(validation_data.shape[0]):
  timestamp_dec0 = calculate_slope_y_intercept(validation_data[i,:,0], time_step, threshold = threshold)
  timestamp_dec1 = calculate_slope_y_intercept(validation_data[i,:,1], time_step, threshold = threshold)
  timestamps_RLED_dec0_Val_list.append(timestamp_dec0)
  timestamps_RLED_dec1_Val_list.append(timestamp_dec1)

for i in range(test_data.shape[0]):
  timestamp_dec0 = calculate_slope_y_intercept(test_data[i,:,0], time_step, threshold = threshold)
  timestamp_dec1 = calculate_slope_y_intercept(test_data[i,:,1], time_step, threshold = threshold)
  timestamps_RLED_dec0_Test_list.append(timestamp_dec0)
  timestamps_RLED_dec1_Test_list.append(timestamp_dec1)


timestamps_RLED_dec0_Train = np.array(timestamps_RLED_dec0_Train_list)
timestamps_RLED_dec0_Val = np.array(timestamps_RLED_dec0_Val_list)
timestamps_RLED_dec0_Test = np.array(timestamps_RLED_dec0_Test_list)

timestamps_RLED_dec1_Train = np.array(timestamps_RLED_dec1_Train_list)
timestamps_RLED_dec1_Val = np.array(timestamps_RLED_dec1_Val_list)
timestamps_RLED_dec1_Test = np.array(timestamps_RLED_dec1_Test_list)

TOF_RLED_Train = timestamps_RLED_dec0_Train - timestamps_RLED_dec1_Train
TOF_RLED_Val = timestamps_RLED_dec0_Val - timestamps_RLED_dec1_Val
TOF_RLED_Test = timestamps_RLED_dec0_Test - timestamps_RLED_dec1_Test

# -------------------------------------------------------------------------
#-------------- EXTRACT WAVEFORMS ACCORDING TO THRESHOLD ------------------
# -------------------------------------------------------------------------

train_data, _, _, _  = extract_signal_along_time(train_data, time_step, fraction = threshold, window_low = window_low, window_high = window_high)
validation_data, _, _, _  = extract_signal_along_time(validation_data, time_step, fraction = threshold, window_low = window_low, window_high = window_high)
test_data, _, _, _  = extract_signal_along_time(test_data, time_step, fraction = threshold, window_low = window_low, window_high = window_high)

# -------------------------------------------------------------------------
#--------------------------- SHIFT BY LED ---------------------------------
# -------------------------------------------------------------------------

train_data = continuous_delay(train_data, TOF_RLED_Train, time_step = time_step, channel_to_fix = 0, channel_to_move = 1)
validation_data = continuous_delay(validation_data, TOF_RLED_Val, time_step = time_step, channel_to_fix = 0, channel_to_move = 1)
test_data = continuous_delay(test_data, TOF_RLED_Test, time_step = time_step, channel_to_fix = 0, channel_to_move = 1)

# -------------------------------------------------------------------------
#-------------------------- CROP WAVEFORM ---------------------------------
# -------------------------------------------------------------------------

train_data = train_data[:,start:stop,:]
validation_data = validation_data[:,start:stop,:] 
test_data = test_data[:,start:stop,:]

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# ---------------------- CALCULATED LED ERROR -----------------------------
# -------------------------------------------------------------------------

REF_train = np.zeros((train_data.shape[0]))
REF_train = 0.0

REF_val = np.zeros((validation_data.shape[0]))
REF_val = 0.0

LED_Error_Train = TOF_RLED_Train - REF_train
LED_Error_Val = TOF_RLED_Val - REF_val

# Put in (N_events, N_channels, N_points)
train_data = np.transpose(train_data, axes = (0, 2, 1))
validation_data = np.transpose(validation_data, axes = (0, 2, 1))
test_data = np.transpose(test_data, axes = (0, 2, 1))

# Create Dataloaders
train_loader = create_dataloaders(train_data, LED_Error_Train, batch_size = batch_size, shuffle = True)
val_loader = create_dataloaders(validation_data, LED_Error_Val, batch_size = batch_size, shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model = ConvolutionalModel(int(stop-start))

print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr) 

# Execute train loop
loss, val_loss = train_loop_convolutional(model, optimizer, train_loader, val_loader, EPOCHS = epochs, name = name,  save = save) 

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
test =  np.squeeze(model(torch.tensor(test_data[:,None,:,:]).to(device).float()).cpu().detach().numpy())

Error_V00_TEST = test[:test_data_55.shape[0]] 
Error_V02_TEST = test[ test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
Error_V20_TEST = test[ test_data_55.shape[0]  + test_data_28.shape[0]:] 

TOF_V00_TEST = TOF_RLED_Test[:test_data_55.shape[0]] 
TOF_V02_TEST = TOF_RLED_Test[ test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]]
TOF_V20_TEST = TOF_RLED_Test[ test_data_55.shape[0]  + test_data_28.shape[0]:] 

TOF_V00 = TOF_V00_TEST - Error_V00_TEST
TOF_V02 = TOF_V02_TEST - Error_V02_TEST
TOF_V20 = TOF_V20_TEST - Error_V20_TEST

# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_V00[None,:], nbins = nbins, limits = 6) 
    
error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + 0.2))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - 0.2))

# Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00),  axis = 1)  
MAE = np.mean(Error, axis = 1)
print(MAE[-1])

# Plot
plt.figure(figsize = (6,4))
plt.plot(np.log10(loss.astype('float32')), label = 'Log Training loss')
plt.plot(np.log10(val_loss.astype('float32')), label = 'Log Validation loss')
plt.ylabel('Logarithmic losses')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Histogram and gaussian fit 
plot_gaussian(TOF_V02, centroid_V00, range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00, centroid_V00, range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20, centroid_V00, range = 0.8, label = ' 0.2 ns offset', nbins = nbins)

params_V02, errors_V02 = get_gaussian_params(TOF_V02, centroid_V00, range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00, centroid_V00, range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20, centroid_V00, range = 0.8, nbins = nbins)

print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()
