import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN


from functions import (extract_signal_along_time_singles, momentos_threshold, set_seed, 
                       normalize, normalize_given_params, continuous_delay, calculate_slope_y_intercept,
                       plot_gaussian, get_gaussian_params, create_and_delay_pulse_pair_along_time)
from Models import  count_parameters
from Train_loops import train_loop_KAN, train_loop_KAN_with_target

def momentos(vector, order = 4):
    
    if  len(np.shape(vector)) == 2:
        Nev, Nt = np.shape(vector)
        t = np.linspace(0, 1, Nt)  # Normalized time array
        MOMENT = np.zeros((Nev, order))
        for i in range(Nev):
                for j in range(order): 
                    W = (t)**(j)
                    MOMENT[i,j] = np.sum(W*vector[i,:])       
   
    
    if  len(np.shape(vector)) == 3:
        Nev, Nt, Nc = np.shape(vector) # Nev: Núm eventos, Nt: Núm puntos temporales, Nc: Número canales
        t = np.linspace(0, 1, Nt)  # Normalized time array
        MOMENT = np.zeros((Nev, order, Nc))
        for i in range(Nev):
            for j in range(order): # Number of moments used
                for k in range(Nc):
                    W = (t)**(j)
                    MOMENT[i,j,k] = np.sum(W*vector[i,:,k])
    
    return MOMENT


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data 
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA.npz'))['data']
data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA.npz'))['data']
data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA.npz'))['data']


# -------------------------------------------------------------------------
# ----------------------- IMPORTANT DEFINITIONS ---------------------------
# -------------------------------------------------------------------------

channel = 1                                # Channel to train
delay_time = 1                             # Max delay to training pulses in ns
moments_order = 5                          # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 71                                 # Num bins for all histograms
t_shift = 1                                # Time steps to move for the new positions
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
epochs = 50                               # Number of epochs for training
lr = 1e-3                                  # Model learning rate
batch_size = 16                            # batch size used for training
total_time = time_step*data_55.shape[1] - time_step  
save = False                                # Save models or not
save_name = 'KAN_models/model_dec' + str(channel)
architecture = [moments_order, 5, 1, 1]   
fraction = 0.1                            # Fraction to trigger the pulse cropping   
window_low = 10                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger



new_train = np.concatenate((data_55[:6000,:,:], data_28[:6000,:,:], data_82[:6000,:,:]), axis = 0) 
new_val = np.concatenate((data_55[6000:7000,:,:], data_28[6000:7000,:,:], data_82[6000:7000,:,:]), axis = 0)
new_test = np.concatenate((data_55[6000:,:,:], data_28[6000:,:,:], data_82[6000:,:,:]), axis = 0)

#print('Número de casos de entrenamiento: ', train_data.shape[0])
#print('Número de casos de validacion: ', validation_data.shape[0])

# -------------------------------------------------------------------------
# ---------------- TRAINING TO REPRODUCE LED ------------------------------
# -------------------------------------------------------------------------
train_array, train_time_array = extract_signal_along_time_singles(new_train[:,:100,channel], time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)
val_array, val_time_array = extract_signal_along_time_singles(new_val[:,:100,channel], time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)
test_array, test_time_array = extract_signal_along_time_singles(new_test[:,:100,channel], time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)

#Get time stamps
timestamps_train_list = []
for i in range(train_array.shape[0]):
  timestamp_train = calculate_slope_y_intercept(new_train[i,:,channel], time_step, threshold = fraction)
  timestamps_train_list.append(timestamp_train)


timestamps_val_list = []
for i in range(new_val.shape[0]):
  timestamp_val = calculate_slope_y_intercept(new_val[i,:,channel], time_step, threshold = fraction)
  timestamps_val_list.append(timestamp_val)


timestamps_test_list = []
for i in range(new_test.shape[0]):
  timestamp_test = calculate_slope_y_intercept(new_test[i,:,channel], time_step, threshold = fraction)
  timestamps_test_list.append(timestamp_test)

timestamps_train = np.array(timestamps_train_list)
timestamps_val = np.array(timestamps_val_list)
timestamps_test = np.array(timestamps_test_list)

# Calculate moments
M_Train = momentos_threshold(train_array, train_time_array, order = moments_order) 
M_Val = momentos_threshold(val_array, val_time_array, order = moments_order) 
MOMENTS_TEST = momentos_threshold(test_array, test_time_array, order = moments_order)


# Normalize moments 
M_Train, params =  normalize(M_Train, method = normalization_method)
M_Val =  normalize_given_params(M_Val[:,:,None], params, channel = 0, method = normalization_method)
MOMENTS_TEST = normalize_given_params(MOMENTS_TEST[:,:,None], params, channel = 0, method = normalization_method)

print("Normalization parameters:", params)

# Create Datasets/Dataloaders
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(M_Train).float(), torch.from_numpy(np.expand_dims(timestamps_train, axis = -1)).float())
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(M_Val).float(), torch.from_numpy(np.expand_dims(timestamps_val, axis = -1)).float())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)

# -------------------------------------------------------------------------
# ------------------------------ MODEL ------------------------------------
# -------------------------------------------------------------------------


model = KAN(architecture)
print(f"Total number of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(model.parameters(), lr = lr) 

loss_target, val_loss_target, test_target, val_target = train_loop_KAN_with_target(model, optimizer, train_loader, val_loader, torch.tensor(MOMENTS_TEST).float(), EPOCHS = epochs, name = save_name, save = save) 

plt.hist(test_target[-1,:] - timestamps_test, bins = nbins)
plt.show()

plt.hist(test_target[-1,:], bins = nbins)
plt.show()

# -------------------------------------------------------------------------
# -------------------- SECOND TRAINING LOOP -------------------------------
# -------------------------------------------------------------------------

# Create virtual coincidences
train, REF_train, time_train = create_and_delay_pulse_pair_along_time(train_array, train_time_array, time_step, total_time, delay_time = delay_time)
val, REF_val, val_time = create_and_delay_pulse_pair_along_time(val_array, val_time_array, time_step, total_time, delay_time = delay_time)


# Calculate moments
M_Train = momentos_threshold(train, time_train, order = moments_order) 
M_Val = momentos_threshold(val, val_time, order = moments_order) 

# Normalize moments 
M_Train, params =  normalize(M_Train, method = normalization_method)
M_Val_channel0 =  normalize_given_params(M_Val, params, channel = 0, method = normalization_method)
M_Val_channel1 =  normalize_given_params(M_Val, params, channel = 1, method = normalization_method)
M_Val = np.stack((M_Val_channel0, M_Val_channel1), axis = -1)

# Create Datasets/Dataloaders
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(M_Train).float(), torch.from_numpy(np.expand_dims(REF_train, axis = -1)).float())
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(M_Val).float(), torch.from_numpy(np.expand_dims(REF_val, axis = -1)).float())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)



# Execute train loop
loss, val_loss, test, val = train_loop_KAN(model, optimizer, train_loader, val_loader, torch.tensor(MOMENTS_TEST).float(), EPOCHS = epochs, name = save_name, save = save) 

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

plt.hist(test[-1,:], bins = nbins, alpha = 0.5, label = 'Detector 0');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Decompress test positions
test_55 = test[-1,:data_55[6000:,:,:].shape[0]]
test_28 = test[-1, data_55[6000:,:,:].shape[0]:data_55[6000:,:,:].shape[0]+ data_28[6000:,:,:].shape[0]]
test_82 = test[-1, data_55[6000:,:,:].shape[0]+ data_28[6000:,:,:].shape[0]:]

MOMENTS_TEST_55 = MOMENTS_TEST[:data_55[6000:,:,:].shape[0], 1]
MOMENTS_TEST_28 = MOMENTS_TEST[data_55[6000:,:,:].shape[0]:data_55[6000:,:,:].shape[0] + data_28[6000:,:,:].shape[0], 1]
MOMENTS_TEST_82 = MOMENTS_TEST[data_55[6000:,:,:].shape[0] + data_28[6000:,:,:].shape[0]:, 1]

# Plot histograms of predictions for the different positions
plt.hist(test_55, bins = nbins, alpha = 0.5, label = '55')
plt.hist(test_28, bins = nbins, alpha = 0.5, label = '28')
plt.hist(test_82, bins = nbins, alpha = 0.5, label = '82')
plt.title('Prediction times histograms')
plt.legend()
plt.show()


#plot validation delays hists
plt.hist(val[-1,:,0] - val[-1,:,1], bins = nbins, alpha = 0.5)
plt.hist(REF_val, bins = nbins, alpha = 0.5)
plt.show()

# Get validation error
err_val = val[-1,:,0] - val[-1,:,1] - REF_val
print('MAE validation: ', np.mean(abs(err_val)))

#Plot validation delay vs error
plt.plot(val[-1,:,0] - val[-1,:,1], err_val, 'b.', markersize = 1.5)
plt.ylabel('Validation Error')
plt.xlabel('Validation target delay')
plt.show()

# Plot moment one vs validation values
plt.plot(MOMENTS_TEST_55, test_55, 'b.', label = '55')
plt.plot(MOMENTS_TEST_28, test_28, 'r.', label = '28')
plt.plot(MOMENTS_TEST_82, test_82, 'g.', label = '82')
plt.xlabel('Moment one')
plt.ylabel('Time (ns)')
plt.legend()
plt.show()

# Plot representation of moments diff vs ref delay
plt.plot(M_Val[:,1,0] - M_Val[:,1,1], REF_val, 'b.')
plt.title('Diff moment vs ref delay')
plt.show()


# Plot Histogram and gaussian fit 
plot_gaussian(err_val, 0.0, range = 0.005, label = 'Validation errors', nbins = nbins)
params, errors = get_gaussian_params(err_val, 0.0, range = 0.005, nbins = nbins)
print("CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params[2], errors[2], params[3], errors[3]))

plt.legend()
plt.xlabel('$\epsilon$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()



