import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import functions
from functions import (create_and_delay_pulse_pair, create_position, 
                       calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, set_seed, continuous_delay)
from Models import ConvolutionalModel,  count_parameters
from Train_loops import train_loop_convolutional

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

delay_time = 1     # Max delay to training pulses in ns
time_step = 0.2     # Signal time step in ns
nbins = 71          # Num bins for all histograms                          
t_shift = 1         # Time steps to move for the new positions
start = 47 
stop = 74 
set_seed(42)        # Fix seeds
epochs = 500
lr = 1e-4
batch_size = 32  
save = False

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

#train_data = new_train[:6000,start:stop,:] 
#validation_data = new_val[:,start:stop,:] 
#test_data = new_train[6000:,start:stop,:]

train_data = np.concatenate((data_55[:2000,start:stop,:], data_28[:2000,start:stop,:], data_82[:2000,start:stop,:]), axis = 0)
validation_data = np.concatenate((data_55[4000:5000,start:stop,:], data_28[4000:5000,start:stop,:], data_82[4000:5000,start:stop,:]), axis = 0)
test_data = np.concatenate((data_55[2000:,start:stop,:], data_28[2000:,start:stop,:], data_82[2000:,start:stop,:]), axis = 0)


print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(train_data[:,:,0], time_step, delay_time = delay_time)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(train_data[:,:,1], time_step, delay_time = delay_time)

val_dec0, REF_val_dec0 = create_and_delay_pulse_pair(validation_data[:,:,0], time_step, delay_time = delay_time)
val_dec1, REF_val_dec1 = create_and_delay_pulse_pair(validation_data[:,:,1], time_step, delay_time = delay_time)

TEST = test_data

# Create Dataset / DataLoaders
train_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec0).float(), torch.from_numpy(np.expand_dims(REF_train_dec0, axis = -1)).float())
train_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec1).float(), torch.from_numpy(np.expand_dims(REF_train_dec1, axis = -1)).float())

val_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(val_dec0).float(), torch.from_numpy(np.expand_dims(REF_val_dec0, axis = -1)).float())
val_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(val_dec1).float(), torch.from_numpy(np.expand_dims(REF_val_dec1, axis = -1)).float())

train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = batch_size, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = batch_size, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = len(val_dataset_dec0), shuffle = False)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = len(val_dataset_dec1), shuffle = False)


# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalModelWithTransformer(nn.Module):
    def __init__(self, N_points, num_outputs=1, num_heads=4, num_transformer_layers=2):
        super(ConvolutionalModelWithTransformer, self).__init__()

        # Define convolutional blocks with batch normalization and ReLU
        self.conv1 = self.conv_block(1, 16, 5)
        self.conv2 = self.conv_block(16, 32, 5)
        self.conv3 = self.conv_block(32, 64, 3)

        # Transformer configuration
        self.embedding_dim = 64  # Transformer expects this as the feature dimension
        self.seq_length = N_points // 8  # Sequence length after convolutional layers

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=num_heads,
                dim_feedforward=128,
                activation='relu',
                dropout=0.1
            ),
            num_layers=num_transformer_layers
        )

        # Fully connected layers
        self.fc1 = nn.Linear(self.seq_length * self.embedding_dim, 32)
        self.fc2 = nn.Linear(32, num_outputs)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.05)

    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same', stride=1),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        # Pass through convolutional blocks
        x = self.conv1(x)  # (batch_size, 16, N_points/2)
        x = self.conv2(x)  # (batch_size, 32, N_points/4)
        x = self.conv3(x)  # (batch_size, 64, N_points/8)

        # Prepare input for the Transformer
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, embedding_dim)
        
        # Pass through Transformer
        x = self.transformer(x)  # (batch_size, seq_length, embedding_dim)

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)  # (batch_size, seq_length * embedding_dim)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softplus(x)

        return x


#set_seed(42)
#model_dec0 = ConvolutionalModel(int(stop-start))
#set_seed(42)
#model_dec1 = ConvolutionalModel(int(stop-start))


set_seed(42)
model_dec0 = ConvolutionalModelWithTransformer(int(stop-start))
set_seed(42)
model_dec1 = ConvolutionalModelWithTransformer(int(stop-start))
print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr, weight_decay = 1e-5) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr, weight_decay = 1e-5) 

#Execute train loop
loss_dec0, test_dec0, val_loss_dec0, val_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/Conv_model_dec0',  save = save) 
loss_dec1, test_dec1, val_loss_dec1, val_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, name = 'predictions/Convolutional/Conv_model_dec1',  save = save)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V00 = TOF[:,:data_55[2000:,:,:].shape[0]] 
TOF_V02 = TOF[:, data_55[2000:,:,:].shape[0] : data_55[2000:,:,:].shape[0] + data_28[2000:,:,:].shape[0]] 
TOF_V20 = TOF[:, data_55[2000:,:,:].shape[0]  + data_28[2000:,:,:].shape[0]:] 
    

# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 
    
error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + t_shift*time_step))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - t_shift*time_step))
#error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 2*t_shift*time_step))
#error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 2*t_shift*time_step))

# Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00),  axis = 1)  #error_V04, error_V40), axis = 1)   
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
plt.hist(test_dec0[-1,:], bins = nbins, range = [-2, 5], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[-1,:], bins = nbins, range = [-2, 5], alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(133)
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Log Training loss Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Log Training loss Detector 1')
plt.plot(np.log10(val_loss_dec0.astype('float32')), label = 'Log Validation loss Detector 0')
plt.plot(np.log10(val_loss_dec1.astype('float32')), label = 'Log Validation loss Detector 1')
plt.ylabel('Logarithmic losses')
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


#print("V40: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
#print("V04: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()


### Combine the two numbers
#num = f"{sys.argv[1]}{sys.argv[2]}"
#num = f"{sys.argv[1]}"
#
## Your existing variables
#FWHM = np.array([params_V02[3], params_V00[3], params_V20[3]])  # ps
#FWHM_err = np.array([errors_V02[3],  errors_V00[3],  errors_V20[3]])        # ps
#centroid = np.array([params_V02[2], params_V00[2], params_V20[2]])  # ps
#centroid_err = np.array([errors_V02[2],  errors_V00[2],  errors_V20[2]])        # ps
#
## Multiply by 1000
#FWHM = FWHM * 1000
#FWHM_err = FWHM_err * 1000
#centroid = centroid * 1000
#centroid_err = centroid_err * 1000
#
#
## Open the file in append mode
#with open('results_FS.txt', 'a') as file:
#    file.write(f"FWHM_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM)}])  # ps\n")
#    file.write(f"FWHM_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in FWHM_err)}])  # ps\n")
#    file.write(f"centroid_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid)}])  # ps\n")
#    file.write(f"centroid_err_{num} = np.array([{', '.join(f'{v:.1f}' for v in centroid_err)}])  # ps\n")
#    file.write(f"MAE_{num} = {MAE[-1]:.7f}  # ps\n")  # Write MAE with one decima
#    file.write("\n")  # Add a new line for better separation