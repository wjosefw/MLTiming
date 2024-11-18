import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import (create_position, momentos_threshold, normalize_given_params, 
                       calculate_gaussian_center, plot_gaussian, get_gaussian_params,
                       continuous_delay, extract_signal_along_time_singles, set_seed)


#Load data
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA.npz'))['data']
data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA.npz'))['data']
data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA.npz'))['data']

test_data  = np.concatenate((data_55[6000:,:,:], data_28[6000:,:,:], data_82[6000:,:,:]), axis = 0)
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

moments_order = 5                          # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 71                                # Num bins for all histograms
t_shift = 1                                # Time steps to move for the new positions
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
total_time = time_step*data_55.shape[1] - time_step 
architecture = [moments_order, 5, 1, 1]   
fraction = 0.01                             # Fraction to trigger the pulse cropping   
window_low = 10                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger
positions = np.array([0.4, 0.2, 0.0, -0.2, -0.4])


# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

test_array_dec0, test_time_array_dec0 = extract_signal_along_time_singles(test_data[:,:100,0], time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)
test_array_dec1, test_time_array_dec1 = extract_signal_along_time_singles(test_data[:,:100,1], time_step, total_time, fraction = fraction, window_low = window_low, window_high = window_high)

test_array = np.stack((test_array_dec0, test_array_dec1), axis = -1)
test_time_array = np.stack((test_time_array_dec0, test_time_array_dec1), axis = -1)

## Calculate moments 
MOMENTS_TEST = momentos_threshold(test_array, test_time_array, order = moments_order)
#
#params_dec1 =  (np.array([3.68875700e+00, 4.82522064e-01, 6.32268770e-02, 8.29974908e-03,
#       1.09150562e-03]), np.array([2.64585153e-01, 3.87722669e-02, 6.11491434e-03, 9.75909816e-04,
#       1.53876365e-04]))
#
#params_dec0 =  (np.array([3.69254424e+00, 4.79422013e-01, 6.23606115e-02, 8.12694738e-03,
#       1.06116437e-03]), np.array([2.46847211e-01, 3.55539483e-02, 5.61759587e-03, 9.02041860e-04,
#       1.42868234e-04]))
#
#MOMENTS_TEST_norm_dec0 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
#MOMENTS_TEST_norm_dec1 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)
#MOMENTS_TEST = np.stack((MOMENTS_TEST_norm_dec0, MOMENTS_TEST_norm_dec1), axis = -1)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/KAN_models'
model_dec0_dir = os.path.join(dir, 'model_dec0')
model_dec1_dir = os.path.join(dir, 'model_dec1')

model_dec0 = KAN(architecture)
model_dec1 = KAN(architecture)

model_dec0.load_state_dict(torch.load(model_dec0_dir))
model_dec1.load_state_dict(torch.load(model_dec1_dir))
model_dec0.eval()
model_dec1.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

test_dec0 = np.squeeze(model_dec0(torch.tensor(MOMENTS_TEST[:,:,0]).float()).detach().numpy())
test_dec1 = np.squeeze(model_dec1(torch.tensor(MOMENTS_TEST[:,:,1]).float()).detach().numpy())

#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/test_dec0_Na22_KAN.npz', data = test_dec0)
#np.savez_compressed('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/test_dec1_Na22_KAN.npz', data = test_dec1)

# Calculate TOF
TOF = test_dec0 - test_dec1


TOF_V00 = TOF[:data_55[6000:,:,:].shape[0]] 
TOF_V02 = TOF[data_55[6000:,:,:].shape[0] : data_55[6000:,:,:].shape[0] + data_28[6000:,:,:].shape[0]] 
TOF_V20 = TOF[data_55[6000:,:,:].shape[0] + data_28[6000:,:,:].shape[0]:] 
    

# Calulate Test error
centroid_V00 = calculate_gaussian_center(TOF_V00[np.newaxis,:], nbins = nbins, limits = 3) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + t_shift*time_step))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - t_shift*time_step))

Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1)
   
# Print MAE
MAE = np.mean(Error, axis = 1)
print(MAE[-1])


# Plot
plt.hist(test_dec0, bins = nbins, range = [0, 3], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, range = [0, 3], alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plot_gaussian(TOF_V02, centroid_V00, range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00, centroid_V00, range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20, centroid_V00, range = 0.8, label = ' 0.2 ns offset', nbins = nbins)

params_V02, errors_V02 = get_gaussian_params(TOF_V02, centroid_V00, range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00, centroid_V00, range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20, centroid_V00, range = 0.8, nbins = nbins)

print("V20: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.3f +/- %.3f  FWHM(ns) = %.3f +/- %.3f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)')
plt.ylabel('Counts')
plt.show()

# -------------------------------------------------------------------------
#--------------------------- ENERGY DEPENDENCE ----------------------------
# -------------------------------------------------------------------------

#dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/'
#energy_dec0 = np.load(os.path.join(dir,'pulsos_Na22_energy_dec0_test_val.npz'), allow_pickle = True)['data']
#energy_dec1 = np.load(os.path.join(dir,'pulsos_Na22_energy_dec1_test_val.npz'), allow_pickle = True)['data']
#
#Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40))   
#MAE = np.mean(Error)
#print('MAE: ', MAE)
#
#plt.plot(energy_dec0 - energy_dec1,  TOF_V00, 'b.', markersize = 1)
#plt.xlabel('Moment 0 diff')
#plt.ylabel('Time difference (ns)')
#plt.show()
#
#
#plt.plot(energy_dec0 - energy_dec1, error_V00, 'b.', markersize = 1)
#plt.xlabel('Energy diff')
#plt.ylabel('Error')
#plt.show()


# -------------------------------------------------------------------------
#--------------------------- BOOTSTRAPING ---------------------------------
# -------------------------------------------------------------------------

resolution_list = []
bias_list = []
MAE_list = []
for i in range(1000):
    a = np.random.choice(np.arange(0, TOF_V00.shape[0]), size = TOF_V00.shape[0], replace = True)
    
    centroid_V00 = calculate_gaussian_center(TOF_V00[None, a], nbins = nbins, limits = 3) 
    params_V04, errors_V04 = get_gaussian_params(TOF_V04[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V02, errors_V02 = get_gaussian_params(TOF_V02[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V00, errors_V00 = get_gaussian_params(TOF_V00[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V20, errors_V20 = get_gaussian_params(TOF_V20[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V40, errors_V40 = get_gaussian_params(TOF_V40[a], centroid_V00, range = 0.8, nbins = nbins)
    
    resolution = np.mean((params_V40[3], params_V20[3], params_V00[3], params_V02[3], params_V04[3]))
    resolution_list.append(resolution)
    
    centroids = np.array([params_V40[2], params_V20[2], params_V00[2], params_V02[2], params_V04[2]])
    bias = np.mean(abs(centroids - positions))
    bias_list.append(bias)

    error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + time_step*t_shift))
    error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
    error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - time_step*t_shift))
    error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 2*time_step*t_shift))
    error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 2*time_step*t_shift))

    Error = np.concatenate((error_V02, error_V00, error_V20, error_V04, error_V40), axis = 1)   
    MAE_list.append(np.mean(Error)) 

print('Mean CTR: ', np.mean(np.array(resolution_list))*1000)
print('Std CTR: ', np.std(np.array(resolution_list))*1000)
print('Mean bias: ', np.mean(np.array(bias_list))*1000)
print('Std bias: ', np.std(np.array(bias_list))*1000)
print('Mean MAE: ', np.mean(np.array(MAE_list))*1000)
print('Std MAE: ', np.std(np.array(MAE_list))*1000)

# -------------------------------------------------------------------------
#-------------------------- INFERENCE TIME --------------------------------
# -------------------------------------------------------------------------

import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

time_test = np.tile(MOMENTS_TEST[0,:,:] , (1000000, 1,1))
model_dec0 = model_dec0.to(device)
time_list_moments = []
time_list_inference = []

# Start timer moments
for i in range(10):
    start_time_momentos = time.time()
    M_time_test = momentos(time_test, order = moments_order)
    end_time_momentos = time.time()
    elapsed_time_momentos = end_time_momentos - start_time_momentos
    time_list_moments.append(elapsed_time_momentos)
time_array_moments = np.array(time_list_moments)


# Start timer inference
for i in range(100):
    start_time_inference= time.time()
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        output_time_test = model_dec0(torch.tensor(M_time_test[:,:,0]).float().to(device))
    end_time_inference = time.time()
    elapsed_time_inference = end_time_inference - start_time_inference
    time_list_inference.append(elapsed_time_inference)
time_array_inference = np.array(time_list_inference)


print('Elapsed time momentos:', np.mean(time_array_moments), np.std(time_array_moments))
print('Elapsed time inference:', np.mean(time_array_inference), np.std(time_array_inference))