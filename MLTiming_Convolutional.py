import numpy as np
import matplotlib.pyplot as plt
import torch

# Data load
V55= np.load('/home/josea/Co60_5cm_5cm.npy')
V28= np.load('/home/josea/Co60_2cm_8cm.npy')
V82= np.load('/home/josea/Co60_8cm_2cm.npy')
REALS = np.concatenate((V28, V55, V82), axis = 0)

data = np.load('/home/josea/PRUEBA_Co60.npz')['data']

# Import functions
from functions import (gauss, gauss_fit, create_and_delay_pulse_pair, create_position, calculate_gaussian_center_sigma, 
                       plot_gaussian_and_get_params, set_seed, interpolate_pulses)
from Models import ConvolutionalModel, train_loop_convolutional
from functions_KAN import count_parameters


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------


delay_steps = 30  # Max number of steps to delay pulses
nbins = 51  # Num bins for all histograms                          
t_shift = 8  # Time steps to move for the new positions
create_positions = True
EXTRASAMPLING = 8
start = 50*EXTRASAMPLING
stop = 74*EXTRASAMPLING
set_seed(42) #Fix seeds
epochs = 500 
lr = 1e-4


# -------------------------------------------------------------------------
#----------------------- INTERPOLATE PULSES -------------------------------
# -------------------------------------------------------------------------

new_data, new_time_step =  interpolate_pulses(data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)
new_REALS, new_time_step =  interpolate_pulses(REALS, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)

# Align the pulses 
align_steps = 20
new_data[:,:,1] = np.roll(new_data[:,:,1], align_steps)
new_data[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)


print('New number of time points: %.d' % (new_data.shape[1]))
print('New time step: %.4f' % (new_time_step))


# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = new_data[:3800,:,:]
test_data = new_data[3800:,:,:]
print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])


# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------


#---------------------------------------------- Move to reference pulse --------------------------------
#mean_pulse_dec0 = get_mean_pulse_from_set(train_data, channel = 0)
#mean_pulse_dec1 = get_mean_pulse_from_set(train_data, channel = 1)

#delays_train_dec0 , moved_pulses_dec0 = move_to_reference(mean_pulse_dec0, train_data, start = start, stop = stop, max_delay = int(stop-start), channel = 0)
#delays_train_dec1 , moved_pulses_dec1 = move_to_reference(mean_pulse_dec1, train_data, start = start, stop = stop, max_delay = int(stop-start), channel = 1)

#train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(moved_pulses_dec0, new_time_step, delay_steps = delay_steps , NOISE = True)
#train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(moved_pulses_dec1, new_time_step, delay_steps = delay_steps , NOISE = True)


#---------------------------------------------- Delay real pulses --------------------------------
train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(train_data[:,start:stop,0], new_time_step, delay_steps = delay_steps, NOISE = True)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(train_data[:,start:stop,1], new_time_step, delay_steps = delay_steps, NOISE = True)

# Create Dataset
train_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec0).float(), torch.from_numpy(np.expand_dims(REF_train_dec0, axis = -1)).float())
train_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(train_dec1).float(), torch.from_numpy(np.expand_dims(REF_train_dec1, axis = -1)).float())

# Create DataLoaders
train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = 32, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = 32, shuffle = True)

#--------------------------------------- Move pulses to a reference one -------------------------------------------------
#delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, validation_data, start = start, stop = stop, max_delay = int(stop-start), channel = 0)
#delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, validation_data, start = start, stop = stop, max_delay = int(stop-start), channel = 1)


#TEST_00 = np.stack((moved_pulses_test_dec0, moved_pulses_test_dec1), axis = 2)
TEST_00 = test_data[:,start:stop,:]
TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift, NOISE = True)
TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift, NOISE = True)  
TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift), NOISE = False)
TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift), NOISE = False)
TEST = np.concatenate((TEST_02, TEST_00, TEST_20, TEST_04, TEST_40), axis = 0)


# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

model_dec0 = ConvolutionalModel(int(stop-start))
model_dec1 = ConvolutionalModel(int(stop-start))

print(f"Total number of parameters: {count_parameters(model_dec0)}")

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr) 

#Execute train loop
loss_dec0, test_dec0 = train_loop_convolutional(model_dec0, optimizer_dec0, train_loader_dec0, torch.tensor(TEST[:,:,0]).float(), EPOCHS = epochs, save = False) 
loss_dec1, test_dec1 = train_loop_convolutional(model_dec1, optimizer_dec1, train_loader_dec1, torch.tensor(TEST[:,:,1]).float(), EPOCHS = epochs, save = False)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

if create_positions == False:
        TOFN_V28 = test_dec0[:,:V28.shape[0]] - test_dec1[:,:V28.shape[0]]
        TOFN_V55 = test_dec0[:,V28.shape[0] :V28.shape[0] + V55.shape[0]] - test_dec1[:,V28.shape[0] :V28.shape[0] + V55.shape[0]]
        TOFN_V82 = test_dec0[:,V28.shape[0] + V55.shape[0]:] - test_dec1[:,V28.shape[0] + V55.shape[0]:]

if create_positions == True:
    TOFN_V02 = test_dec0[:,:TEST_00.shape[0]] - test_dec1[:,:TEST_00.shape[0]]
    TOFN_V00 = test_dec0[:,TEST_00.shape[0] : 2*TEST_00.shape[0]] - test_dec1[:, TEST_00.shape[0] : 2*TEST_00.shape[0]]
    TOFN_V20 = test_dec0[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]] - test_dec1[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]]
    TOFN_V04 = test_dec0[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]] - test_dec1[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]]
    TOFN_V40 = test_dec0[:,4*TEST_00.shape[0]:] - test_dec1[:,4*TEST_00.shape[0]:]

# Calulate Validation error
if create_positions == False:
    
    # Calculate centered position 'centroid'
    centroid_V55, sigmaN_V55 = calculate_gaussian_center_sigma(TOFN_V55, np.zeros((TOFN_V55.shape[0])),  nbins = nbins)  
    
    error_V28 = abs((TOFN_V28 - centroid_V55[:, np.newaxis] + 0.2))
    error_V55 = abs((TOFN_V55 - centroid_V55[:, np.newaxis]))
    error_V82 = abs((TOFN_V82 - centroid_V55[:, np.newaxis] - 0.2))
    Error = np.concatenate((error_V28, error_V55, error_V82), axis = 1)

if create_positions == True:
    # Calculate centered position 'centroid'
    centroid_V00, sigmaN_V00 = calculate_gaussian_center_sigma(TOFN_V00, np.zeros((TOFN_V00.shape[0])), nbins = nbins) 
    
    error_V02 = abs((TOFN_V02 - centroid_V00[:, np.newaxis] + 0.2))
    error_V00 = abs((TOFN_V00 - centroid_V00[:, np.newaxis]))
    error_V20 = abs((TOFN_V20 - centroid_V00[:, np.newaxis] - 0.2))
    error_V04 = abs((TOFN_V04 - centroid_V00[:, np.newaxis] + 0.4))
    error_V40 = abs((TOFN_V40 - centroid_V00[:, np.newaxis] - 0.4))

    Error = np.concatenate((error_V02, error_V00, error_V20, error_V04, error_V40), axis = 1)
    

# Print MAE
MAE = np.mean(Error, axis = 1)
idx_min_MAE = np.where(MAE == np.min(MAE))[0][0]
print(idx_min_MAE, np.min(MAE))


# PLot
plt.figure(figsize = (20,5))
plt.subplot(131)
plt.plot(np.log10(MAE.astype('float64')), label = 'MAE')
plt.title('Results in coincidence')
plt.xlabel('Epochs')
plt.ylabel('Log10')
plt.legend()

plt.subplot(132)
plt.hist(test_dec0[idx_min_MAE,:], bins = nbins, range = [-1, 5], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1[idx_min_MAE,:], bins = nbins, range = [-1, 5], alpha = 0.5, label = 'Detector 1');
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
HN, AN, x0N_V04, sigmaN_V04, FWHMN_V04 = plot_gaussian_and_get_params(TOFN_V04[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = '-0.4 ns offset', nbins = nbins)
HN, AN, x0N_V02, sigmaN_V02, FWHMN_V02 = plot_gaussian_and_get_params(TOFN_V02[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
HN, AN, x0N_V00, sigmaN_V00, FWHMN_V00 = plot_gaussian_and_get_params(TOFN_V00[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
HN, AN, x0N_V20, sigmaN_V20, FWHMN_V20 = plot_gaussian_and_get_params(TOFN_V20[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
HN, AN, x0N_V40, sigmaN_V40, FWHMN_V40 = plot_gaussian_and_get_params(TOFN_V40[idx_min_MAE,:], centroid_V00[idx_min_MAE], range = 0.8, label = ' 0.4 ns offset', nbins = nbins)

print('')
print("V40: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V40, FWHMN_V40, sigmaN_V40))
print("V02: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V02, FWHMN_V02, sigmaN_V02))
print("V00: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V00, FWHMN_V00, sigmaN_V00))
print("V20: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V20, FWHMN_V20, sigmaN_V20))
print("V04: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V04, FWHMN_V04, sigmaN_V04))

plt.legend()
plt.xlabel('$\Delta t$ (ns)')
plt.ylabel('Counts')
plt.show()
