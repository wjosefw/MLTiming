import os
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Import functions
from functions import (create_and_delay_pulse_pair, set_seed, interpolate_pulses, plot_gaussian, 
                       get_gaussian_params, create_position, calculate_gaussian_center_sigma)
from Models import MLP_Torch, train_loop_MLP
from functions_KAN import count_parameters

# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'

train_data = np.load(os.path.join(dir,'Na22_train.npz'))['data']
val_data = np.load(os.path.join(dir, 'Na22_val.npz'))['data']
test_data = np.load(os.path.join(dir, 'Na22_test_val.npz'))['data']



# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_steps = 30        # Max number of steps to delay pulses
set_seed(42)            # Fix seeds
nbins = 91              # Num bins for all histograms
t_shift = 8             # Time steps to move for the new positions
EXTRASAMPLING = 8
start = 50*EXTRASAMPLING 
stop = 74*EXTRASAMPLING 
lr = 1e-3
epochs = 500
Num_Neurons = 64

# -------------------------------------------------------------------------
#----------------------- INTERPOLATE PULSES -------------------------------
# -------------------------------------------------------------------------

new_train, new_time_step =  interpolate_pulses(train_data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)
new_val, new_time_step =  interpolate_pulses(val_data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)
new_test, new_time_step =  interpolate_pulses(test_data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)

# Align the pulses 
align_steps = 20

new_train[:,:,1] = np.roll(new_train[:,:,1], align_steps)
new_val[:,:,1] = np.roll(new_val[:,:,1], align_steps)
new_test[:,:,1] = np.roll(new_test[:,:,1], align_steps)

new_train[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)
new_val[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)
new_test[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)

print('New number of time points: %.d' % (new_train.shape[1]))
print('New time step: %.4f' % (new_time_step))

# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = new_train[:,start:stop,:]
validation_data = new_val[:,start:stop,:]
test_data = new_test[:,start:stop,:]
print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------


train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(train_data[:,:,0], new_time_step, delay_steps = delay_steps, NOISE = False)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(train_data[:,:,1], new_time_step, delay_steps = delay_steps, NOISE = False)

val_dec0, REF_val_dec0 = create_and_delay_pulse_pair(validation_data[:,:,0], new_time_step, delay_steps = delay_steps, NOISE = False)
val_dec1, REF_val_dec1 = create_and_delay_pulse_pair(validation_data[:,:,1], new_time_step, delay_steps = delay_steps, NOISE = False)


# Test set
TEST_00 = test_data 
TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift, NOISE = False)
TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift, NOISE = False)
TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift), NOISE = False)
TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift), NOISE = False)
TEST = np.concatenate((TEST_02, TEST_00, TEST_20, TEST_04, TEST_40), axis = 0)


# Create Datasets/Dataloaders
train_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec0).float(), torch.from_numpy(np.expand_dims(REF_train_dec0, axis = -1)).float())
train_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec1).float(), torch.from_numpy(np.expand_dims(REF_train_dec1, axis = -1)).float())

val_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(val_dec0).float(), torch.from_numpy(np.expand_dims(REF_val_dec0, axis = -1)).float())
val_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(val_dec1).float(), torch.from_numpy(np.expand_dims(REF_val_dec1, axis = -1)).float())

train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = 32, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = 32, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = 32, shuffle = True)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = 32, shuffle = True)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model_dec0 = MLP_Torch(NM = int(stop-start), NN = Num_Neurons, STD_INIT = 0.5)
model_dec1 = MLP_Torch(NM = int(stop-start), NN = Num_Neurons, STD_INIT = 0.5)
         
print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr)  

# Execute train loop
loss_dec0, val_loss_dec0, test_dec0 = train_loop_MLP(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, save = False) 
loss_dec1, val_loss_dec1, test_dec1 = train_loop_MLP(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, save = False)


# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------


TOF_V02 = test_dec0[:,:TEST_00.shape[0]] - test_dec1[:,:TEST_00.shape[0]]
TOF_V00 = test_dec0[:,TEST_00.shape[0] : 2*TEST_00.shape[0]] - test_dec1[:,TEST_00.shape[0] : 2*TEST_00.shape[0]]
TOF_V20 = test_dec0[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]] - test_dec1[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]]
TOF_V04 = test_dec0[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]] - test_dec1[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]]
TOF_V40 = test_dec0[:,4*TEST_00.shape[0]:] - test_dec1[:,4*TEST_00.shape[0]:]
    

# Calculate centered position 'centroid'
centroid_V00, sigmaN_V00 = calculate_gaussian_center_sigma(TOF_V00, np.zeros((TOF_V00.shape[0])), nbins = nbins) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + 0.2))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - 0.2))
error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 0.4))
error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 0.4))

#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40), axis = 1)   
MAE = np.mean(Error, axis = 1)
print(MAE[-1])

# PLot
plt.figure(figsize = (20,5))
plt.subplot(131)
plt.plot(np.log10(MAE.astype('float64')), label = 'MAE')
plt.title('Results in coincidence')
plt.xlabel('Epochs')
plt.ylabel('Log10')
plt.legend()

plt.subplot(132)
plt.hist(test_dec0[-1,:], bins = nbins, range = [-1, 5], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[-1,:], bins = nbins, range = [-1, 5], alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(133)
plt.plot(np.log10(loss_dec0.astype('float32')), label = 'Detector 0')
plt.plot(np.log10(loss_dec1.astype('float32')), label = 'Detector 1')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

  
# Histogram and gaussian fit 
plot_gaussian(TOF_V04[-1,:], centroid_V00[-1], range = 0.8, label = '-0.4 ns offset', nbins = nbins)
plot_gaussian(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V40[-1,:], centroid_V00[-1], range = 0.8, label = ' 0.4 ns offset', nbins = nbins)


params_V04, errors_V04 = get_gaussian_params(TOF_V04[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V02, errors_V02 = get_gaussian_params(TOF_V02[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)
params_V40, errors_V40 = get_gaussian_params(TOF_V40[-1,:], centroid_V00[-1], range = 0.8, nbins = nbins)


print("V40: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
print("V04: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

