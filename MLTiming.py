import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from efficient_kan.src.efficient_kan import KAN
#from faster_kan.fastkan.fastkan import FastKAN
#from kan import *
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Import functions 
from functions import momentos, create_and_delay_pulse_pair, create_position, set_seed, calculate_gaussian_center_sigma, normalize, normalize_given_params, plot_gaussian_and_get_params, interpolate_pulses
from functions_KAN import  count_parameters, train_loop_KAN


# Load data 
V55 = np.load('/home/josea/Co60_5cm_5cm.npy')
V28 = np.load('/home/josea/Co60_2cm_8cm.npy')
V82 = np.load('/home/josea/Co60_8cm_2cm.npy')
REALS = np.concatenate((V28, V55, V82), axis = 0)

data = np.load('/home/josea/PRUEBA_Co60.npz')['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

delay_steps = 30        # Max number of steps to delay pulses
moments_order = 6       # Max order of moments used
set_seed(42)            # Fix seeds
nbins = 51              # Num bins for all histograms
create_positions = 0    # Wether to create new_source positions. (0 = YES, 1 = NO)                           
t_shift = 8             # Time steps to move for the new positions
normalization_method = 'min-max'
EXTRASAMPLING = 8
start = 50*EXTRASAMPLING 
stop = 74*EXTRASAMPLING 
lr = 1e-3
epochs = 500

# -------------------------------------------------------------------------
#----------------------- INTERPOLATE PULSES -------------------------------
# -------------------------------------------------------------------------

new_data, new_time_step =  interpolate_pulses(data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)
new_REALS, new_time_step =  interpolate_pulses(REALS, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)

# Align the pulses 
align_steps = 20
new_data[:,:,1] = np.roll(new_data[:,:,1], align_steps)
new_data[:,:align_steps,1] = np.random.normal(scale = 1e-3, size = align_steps)


print('New number of time points: %.d' % (new_data.shape[1]))
print('New time step: %.4f' % (new_time_step))


# -------------------------------------------------------------------------
#----------------------- TRAIN/TEST SPLIT ---------------------------------
# -------------------------------------------------------------------------

train_data = new_data[:3600,:,:]
validation_data = new_data[3600:3800,:,:]
test_data = new_data[3800:,:,:]
print('Número de casos de entrenamiento: ', train_data.shape[0])
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# -------------------- TRAIN/VALIDATION/TEST SET --------------------------
# -------------------------------------------------------------------------


train_dec0, REF_train_dec0 = create_and_delay_pulse_pair(train_data[:,start:stop,0], new_time_step, delay_steps = delay_steps, NOISE = False)
train_dec1, REF_train_dec1 = create_and_delay_pulse_pair(train_data[:,start:stop,1], new_time_step, delay_steps = delay_steps, NOISE = False)

val_dec0, REF_val_dec0 = create_and_delay_pulse_pair(validation_data[:,start:stop,0], new_time_step, delay_steps = delay_steps, NOISE = False)
val_dec1, REF_val_dec1 = create_and_delay_pulse_pair(validation_data[:,start:stop,1], new_time_step, delay_steps = delay_steps, NOISE = False)

if create_positions == 1:
    TEST = test_data[:,start:stop,:]

if create_positions == 0:
    TEST_00 = test_data[:,start:stop,:] 
    TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift, NOISE = False)
    TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift, NOISE = False)
    TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift), NOISE = False)
    TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift), NOISE = False)
    TEST = np.concatenate((TEST_02, TEST_00, TEST_20, TEST_04, TEST_40), axis = 0)


# Calculate moments 
M_Train_dec0 = momentos(train_dec0, order = moments_order) 
M_Train_dec1 = momentos(train_dec1, order = moments_order) 
M_Train_dec0, params_dec0 =  normalize(M_Train_dec0, method = normalization_method)
M_Train_dec1, params_dec1 =  normalize(M_Train_dec1, method = normalization_method)

M_Val_dec0 = momentos(val_dec0, order = moments_order) 
M_Val_dec1 = momentos(val_dec1, order = moments_order) 

M_Val_dec0_channel0 =  normalize_given_params(M_Val_dec0, params_dec0, channel = 0, method = normalization_method)
M_Val_dec0_channel1 =  normalize_given_params(M_Val_dec0, params_dec0, channel = 1, method = normalization_method)
M_Val_dec0 = np.stack((M_Val_dec0_channel0, M_Val_dec0_channel1), axis = -1)

M_Val_dec1_channel0 =  normalize_given_params(M_Val_dec1, params_dec1, channel = 0, method = normalization_method)
M_Val_dec1_channel1 =  normalize_given_params(M_Val_dec1, params_dec1, channel = 1, method = normalization_method)
M_Val_dec1 = np.stack((M_Val_dec1_channel0, M_Val_dec1_channel1), axis = -1)


MOMENTS_TEST = momentos(TEST, order = moments_order)
MOMENTS_TEST_norm_dec0 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
MOMENTS_TEST_norm_dec1 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)
MOMENTS_TEST = np.stack((MOMENTS_TEST_norm_dec0, MOMENTS_TEST_norm_dec1), axis = -1)


# Print information 
print(M_Train_dec0.shape, "NM dec0 =", M_Train_dec0.shape[1])
print(M_Train_dec1.shape, "NM dec1 =", M_Train_dec1.shape[1])
print("Normalization parameters detector 0:", params_dec0)
print("Normalization parameters detector 1:", params_dec1)


# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------

NM = M_Train_dec0.shape[1]
architecture = [NM, NM*2, 1, 1]   
grid_size = 5   
k_order = 3

model_dec0 = KAN(architecture)
model_dec1 = KAN(architecture)
         
print(f"Total number of parameters: {count_parameters(model_dec0)}")

# Create Dataset
train_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(M_Train_dec0).float(), torch.from_numpy(np.expand_dims(REF_train_dec0, axis = -1)).float())
train_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(M_Train_dec1).float(), torch.from_numpy(np.expand_dims(REF_train_dec1, axis = -1)).float())

val_dataset_dec0 = torch.utils.data.TensorDataset(torch.from_numpy(M_Val_dec0).float(), torch.from_numpy(np.expand_dims(REF_val_dec0, axis = -1)).float())
val_dataset_dec1 = torch.utils.data.TensorDataset(torch.from_numpy(M_Val_dec1).float(), torch.from_numpy(np.expand_dims(REF_val_dec1, axis = -1)).float())

# Create DataLoaders
train_loader_dec0 = torch.utils.data.DataLoader(train_dataset_dec0, batch_size = 32, shuffle = True)
train_loader_dec1 = torch.utils.data.DataLoader(train_dataset_dec1, batch_size = 32, shuffle = True)

val_loader_dec0 = torch.utils.data.DataLoader(val_dataset_dec0, batch_size = 32, shuffle = True)
val_loader_dec1 = torch.utils.data.DataLoader(val_dataset_dec1, batch_size = 32, shuffle = True)

optimizer_dec0 = torch.optim.AdamW(model_dec0.parameters(), lr = lr) 
optimizer_dec1 = torch.optim.AdamW(model_dec1.parameters(), lr = lr)  

# Execute train loop
loss_dec0, test_dec0 = train_loop_KAN(model_dec0, optimizer_dec0, train_loader_dec0, val_loader_dec0, torch.tensor(MOMENTS_TEST[:,:,0]).float(), EPOCHS = epochs, save = False) 
loss_dec1, test_dec1 = train_loop_KAN(model_dec1, optimizer_dec1, train_loader_dec1, val_loader_dec1, torch.tensor(MOMENTS_TEST[:,:,1]).float(), EPOCHS = epochs, save = False)

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

if create_positions == 1:
    
    TOFN_V28 = test_dec0[:,:V28.shape[0]] - test_dec1[:,:V28.shape[0]]
    TOFN_V55 = test_dec0[:,V28.shape[0] :V28.shape[0] + V55.shape[0]] - test_dec1[:,V28.shape[0] :V28.shape[0] + V55.shape[0]]
    TOFN_V82 = test_dec0[:,V28.shape[0] + V55.shape[0]:] - test_dec1[:,V28.shape[0] + V55.shape[0]:]

if create_positions == 0:
    
    TOFN_V02 = test_dec0[:,:TEST_00.shape[0]] - test_dec1[:,:TEST_00.shape[0]]
    TOFN_V00 = test_dec0[:,TEST_00.shape[0] : 2*TEST_00.shape[0]] - test_dec1[:,TEST_00.shape[0] : 2*TEST_00.shape[0]]
    TOFN_V20 = test_dec0[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]] - test_dec1[:,2*TEST_00.shape[0] :3*TEST_00.shape[0]]
    TOFN_V04 = test_dec0[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]] - test_dec1[:,3*TEST_00.shape[0] :4*TEST_00.shape[0]]
    TOFN_V40 = test_dec0[:,4*TEST_00.shape[0]:] - test_dec1[:,4*TEST_00.shape[0]:]
    

# Calulate Validation error
if create_positions == 1:
    
    # Calculate centered position 'centroid'
    centroid_V55, sigmaN_V55 = calculate_gaussian_center_sigma(TOFN_V55, np.zeros((TOFN_V55.shape[0])), nbins = nbins)  
    
    error_V28 = abs((TOFN_V28 - centroid_V55[:, np.newaxis] + 0.2))
    error_V55 = abs((TOFN_V55 - centroid_V55[:, np.newaxis]))
    error_V82 = abs((TOFN_V82 - centroid_V55[:, np.newaxis] - 0.2))
    Error = np.concatenate((error_V28, error_V55, error_V82), axis = 1)

if create_positions == 0:

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

epoch = idx_min_MAE  
# Histogram and gaussian fit 
if create_positions == 0:
    HN, AN, x0N_V04, sigmaN_V04, FWHMN_V04 = plot_gaussian_and_get_params(TOFN_V04[epoch,:], centroid_V00[epoch], range = 0.8, label = '-0.4 ns offset', nbins = nbins)
    HN, AN, x0N_V28, sigmaN_V28, FWHMN_V28 = plot_gaussian_and_get_params(TOFN_V02[epoch,:], centroid_V00[epoch], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
    HN, AN, x0N_V55, sigmaN_V55, FWHMN_V55 = plot_gaussian_and_get_params(TOFN_V00[epoch,:], centroid_V00[epoch], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
    HN, AN, x0N_V82, sigmaN_V82, FWHMN_V82 = plot_gaussian_and_get_params(TOFN_V20[epoch,:], centroid_V00[epoch], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
    HN, AN, x0N_V40, sigmaN_V40, FWHMN_V40 = plot_gaussian_and_get_params(TOFN_V40[epoch,:], centroid_V00[epoch], range = 0.8, label = ' 0.4 ns offset', nbins = nbins)

    print('')
    print("V40: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V40, FWHMN_V40, sigmaN_V40))
    print("V82: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V82, FWHMN_V82, sigmaN_V82))
    print("V55: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V55, FWHMN_V55, sigmaN_V55))
    print("V28: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V28, FWHMN_V28, sigmaN_V28))
    print("V04: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V04, FWHMN_V04, sigmaN_V04))

if create_positions == 1:
    HN, AN, x0N_V28, sigmaN_V28, FWHMN_V28 = plot_gaussian_and_get_params(TOFN_V28[epoch,:], centroid_V55[epoch], range = 0.8, label = '-0.2 ns offset', nbins = nbins)
    HN, AN, x0N_V55, sigmaN_V55, FWHMN_V55 = plot_gaussian_and_get_params(TOFN_V55[epoch,:], centroid_V55[epoch], range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
    HN, AN, x0N_V82, sigmaN_V82, FWHMN_V82 = plot_gaussian_and_get_params(TOFN_V82[epoch,:], centroid_V55[epoch], range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
   

    print('')
    print("V82: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V82, FWHMN_V82, sigmaN_V82))
    print("V55: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V55, FWHMN_V55, sigmaN_V55))
    print("V28: CENTROID(ns) = %.3f  FWHM(ns) = %.3f  std(ns) = %.3f" % (x0N_V28, FWHMN_V28, sigmaN_V28))

plt.legend()
plt.xlabel('$\Delta t$ (ns)')
plt.ylabel('Counts')
plt.show()
