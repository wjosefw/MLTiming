import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import functions
from functions import (calculate_gaussian_center, plot_gaussian, get_gaussian_params, set_seed)
from Models import ConvolutionalModel, count_parameters
from Train_loops import train_loop_convolutional


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

        # Define convolutional blocks 
        self.conv1 = self.conv_block1(1, 16, 5) 
        self.conv2 = self.conv_block2(16, 32, 5) 
        self.conv3 = self.conv_block2(32, 64, 3)
        

        # Calculate flattened size for fully connected layer
        self.flatten_size = 64 * (N_points // 8)  

        # Fully connected layer with output
        self.fc1 = nn.Linear(self.flatten_size, 16)
        self.fc2 = nn.Linear(16, num_outputs)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p = 0.05)

    def conv_block1(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = 'same', stride = 1),
            nn.MaxPool2d(kernel_size = (2,2)),
        )
    
    def conv_block2(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, padding = 'same', stride = 1),
            nn.MaxPool1d(kernel_size = 2),
        )


    def forward(self, x):
        # Pass through convolutional blocks
        x = self.conv1(x)
        x = x.squeeze(2) 
        x = self.conv2(x)
        x = self.conv3(x)
     
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        

        return x

def train_loop_convolutional(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_tensor = test_tensor.to(device)

    loss_fn = torch.nn.L1Loss()  # Define MAE loss

    loss_list = []
    val_loss_list = []
    test = []
    val_list = []

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
            test_epoch = model(test_tensor[:,None,:,:])
            test.append(np.squeeze(test_epoch.cpu().numpy()))

            val_loss = 0.0
            val_stack = []  # List to hold val_0 and val_1 pairs

            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                mask = val_labels != 0
                val = model(val_data[:,None,:,:])
                val_loss += loss_fn(val[mask], val_labels[mask]).item()

                # Stack val predictions along the last dimension
                val_stack.append(np.stack((np.squeeze(val.cpu().detach().numpy()))))

            # Combine all batches into a single array for this epoch
            epoch_val = np.concatenate(val_stack, axis=0)
            val_list.append(epoch_val)

            val_loss_list.append(val_loss / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

    if save:
        torch.save(model.state_dict(), name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype='object')
    test = np.array(test, dtype='object')
    val_loss = np.array(val_loss_list, dtype='object')
    val = np.array(val_list, dtype='object')

    return loss_array, test, val_loss, val


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

time_step = 0.2     # Signal time step in ns
nbins = 71          # Num bins for all histograms                          
start = 47 
stop =  74 
set_seed(42)        # Fix seeds
epochs = 500
lr = 1e-4
batch_size = 32  
save = False

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = np.concatenate((train_data_55, train_data_28, train_data_82), axis = 0)
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

train_data = train_data[:,start:stop,:]
validation_data = validation_data[:,start:stop,:] 
test_data = test_data[:,start:stop,:]

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

REF_train = np.zeros((train_data.shape[0]))
REF_train[:train_data_55.shape[0]] = 0.0
REF_train[train_data_55.shape[0]:train_data_55.shape[0] + train_data_28.shape[0]] = -0.2
REF_train[train_data_55.shape[0] + train_data_28.shape[0]:] = 0.2 

REF_val = np.zeros((validation_data.shape[0]))
REF_val[:validation_data_55.shape[0]] = 0.0
REF_val[validation_data_55.shape[0]:validation_data_55.shape[0] + validation_data_28.shape[0]] = -0.2
REF_val[validation_data_55.shape[0] + validation_data_28.shape[0]:] = 0.2 

train_data = np.transpose(train_data, axes=(0, 2, 1))
validation_data = np.transpose(validation_data, axes=(0, 2, 1))
test_data = np.transpose(test_data, axes=(0, 2, 1))

print(train_data.shape)

# Create Dataset / DataLoaders
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(np.expand_dims(REF_train, axis = -1)).float())
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(validation_data).float(), torch.from_numpy(np.expand_dims(REF_val, axis = -1)).float())


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model = ConvolutionalModel(int(stop-start))

print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-5) 


#Execute train loop
loss, test, val_loss, val = train_loop_convolutional(model, optimizer, train_loader, val_loader, torch.tensor(test_data).float(), EPOCHS = epochs, name = 'predictions/Convolutional/model_dec0',  save = save) 

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

# Calculate TOF

TOF_V00 = test[:,:test_data_55.shape[0]] 
TOF_V02 = test[:, test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
TOF_V20 = test[:, test_data_55.shape[0]  + test_data_28.shape[0]:] 
    

# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_V00, nbins = nbins, limits = 3) 
    
error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + 0.2))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - 0.2))


# Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00),  axis = 1)  
MAE = np.mean(Error, axis = 1)
print(MAE[-1])


# Plot
plt.figure(figsize = (20,5))
plt.subplot(121)
plt.plot(np.log10(MAE.astype('float64')), label = 'MAE')
plt.title('Results in coincidence')
plt.xlabel('Epochs')
plt.ylabel('Log10')
plt.legend()

plt.subplot(122)
plt.plot(np.log10(loss.astype('float32')), label = 'Log Training loss')
plt.plot(np.log10(val_loss.astype('float32')), label = 'Log Validation loss')
plt.ylabel('Logarithmic losses')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plot_gaussian(TOF_V02[-1,:], centroid_V00[-1], range = 0.25, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00[-1,:], centroid_V00[-1], range = 0.25, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20[-1,:], centroid_V00[-1], range = 0.25, label = ' 0.2 ns offset', nbins = nbins)


params_V02, errors_V02 = get_gaussian_params(TOF_V02[-1,:], centroid_V00[-1], range = 0.25, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00[-1,:], centroid_V00[-1], range = 0.25, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20[-1,:], centroid_V00[-1], range = 0.25, nbins = nbins)


print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))


print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()


