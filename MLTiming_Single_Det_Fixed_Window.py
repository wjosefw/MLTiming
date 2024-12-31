import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN


from functions import (set_seed, plot_gaussian, 
                       get_gaussian_params, create_and_delay_pulse_pair)
from Models import  count_parameters, ConvolutionalModel
from Train_loops import train_loop_convolutional


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
epochs = 500                               # Number of epochs for training
lr = 1e-4                                  # Model learning rate
batch_size = 32                            # batch size used for training
save = False                               # Save models or not
save_name = 'predictions/Convolutional/Conv_model_dec' + str(channel)

# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------

train_data = np.concatenate((train_data_55, train_data_28, train_data_82), axis = 0)
validation_data = np.concatenate((validation_data_55, validation_data_28, validation_data_82), axis = 0)
test_data = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)

train_data = train_data[:,start:stop,:]
validation_data = validation_data[:,start:stop,:] 
test_data = test_data[:,start:stop,:]

print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de validacion: ', validation_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------

# Create virtual coincidences
train, REF_train = create_and_delay_pulse_pair(train_data[:,:,channel], time_step, delay_time = delay_time)
val, REF_val = create_and_delay_pulse_pair(validation_data[:,:,channel], time_step, delay_time = delay_time)

TEST = test_data

# Create Datasets/Dataloaders
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train).float(), torch.from_numpy(np.expand_dims(REF_train, axis = -1)).float())
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val).float(), torch.from_numpy(np.expand_dims(REF_val, axis = -1)).float())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

set_seed(42)
model = ConvolutionalModel(int(stop-start))

print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-5) 

#Execute train loop
loss, test, val_loss, val_inf = train_loop_convolutional(model, optimizer, train_loader, val_loader, torch.tensor(TEST[:,:,channel]).float(), EPOCHS = epochs, name = save_name,  save = save) 


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

plt.hist(test[-1,:], bins = nbins, alpha = 0.5, range = [0, 6], label = 'Detector 0');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Decompress test positions
test_55 = test[-1,:test_data_55.shape[0]]
test_28 = test[-1, test_data_55.shape[0]: test_data_55.shape[0]+ test_data_28.shape[0]]
test_82 = test[-1, test_data_55.shape[0]+ test_data_28.shape[0]:]


# Plot histograms of predictions for the different positions
plt.hist(test_55, bins = nbins, alpha = 0.5, label = '55')
plt.hist(test_28, bins = nbins, alpha = 0.5, label = '28')
plt.hist(test_82, bins = nbins, alpha = 0.5, label = '82')
plt.title('Prediction times histograms')
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

#Plot validation delay vs error
plt.plot(val_inf[-1,:,0] - val_inf[-1,:,1], err_val, 'b.', markersize = 1.5)
plt.ylabel('Validation Error')
plt.xlabel('Validation target delay')
plt.show()


# Plot Histogram and gaussian fit 
plot_gaussian(err_val, 0.0, range = 0.1, label = 'Validation errors', nbins = nbins)
params, errors = get_gaussian_params(err_val, 0.0, range = 0.1, nbins = nbins)
print("CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params[2], errors[2], params[3], errors[3]))

plt.legend()
plt.xlabel('$\epsilon$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()



