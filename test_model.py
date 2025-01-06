import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import (momentos_threshold, normalize_given_params, 
                       calculate_gaussian_center, plot_gaussian, get_gaussian_params,
                       extract_signal_along_time_singles, set_seed)



#Load data
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
test_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_test.npz'))['data']
test_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_test.npz'))['data']
test_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_test.npz'))['data']


test_data  = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

moments_order = 7                         # Order of moments used
set_seed(42)                               # Fix seeds
nbins = 71                                # Num bins for all histograms
normalization_method = 'standardization'
time_step = 0.2                            # Signal time step in ns
architecture = [moments_order, 3, 1, 1]   
fraction = 0.1                             # Fraction to trigger the pulse cropping   
window_low = 14                            # Number of steps to take before trigger
window_high = 10                           # Number of steps to take after trigger
positions = np.array([-0.2, 0.0, 0.2])


# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

test_array_dec0, test_time_array_dec0 = extract_signal_along_time_singles(test_data[:,:100,0], time_step, fraction = fraction, window_low = window_low, window_high = window_high)
test_array_dec1, test_time_array_dec1 = extract_signal_along_time_singles(test_data[:,:100,1], time_step, fraction = fraction, window_low = window_low, window_high = window_high)

test_array = np.stack((test_array_dec0, test_array_dec1), axis = -1)
test_time_array = np.stack((test_time_array_dec0, test_time_array_dec1), axis = -1)

## Calculate moments 
MOMENTS_TEST = momentos_threshold(test_array, test_time_array, order = moments_order)

params_dec0 = (np.array([3.09991389e+00, 4.63085059e+01, 6.92581996e+02, 1.03715011e+04,
       1.55529650e+05, 2.33567496e+06, 3.51281042e+07]), np.array([2.16221785e-01, 3.46893863e+00, 5.92896327e+01, 1.03886921e+03,
       1.82331281e+04, 3.17505692e+05, 5.47156791e+06]))

params_dec1 = (np.array([3.09495111e+00, 4.66087007e+01, 7.02612076e+02, 1.06039732e+04,
       1.60241230e+05, 2.42473896e+06, 3.67418671e+07]), np.array([2.31959436e-01, 3.79617696e+00, 6.52555929e+01, 1.14135496e+03,
       1.99603585e+04, 3.46564142e+05, 5.96232795e+06]))


MOMENTS_TEST_norm_dec0 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
MOMENTS_TEST_norm_dec1 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)
MOMENTS_TEST = np.stack((MOMENTS_TEST_norm_dec0, MOMENTS_TEST_norm_dec1), axis = -1)


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
#model_dec0.eval()
#model_dec1.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

test_dec0 = np.squeeze(model_dec0(torch.tensor(MOMENTS_TEST[:,:,0]).float()).detach().numpy())
test_dec1 = np.squeeze(model_dec1(torch.tensor(MOMENTS_TEST[:,:,1]).float()).detach().numpy())

# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V00 = TOF[:test_data_55.shape[0]] 
TOF_V02 = TOF[test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
TOF_V20 = TOF[test_data_55.shape[0] + test_data_28.shape[0]:] 


# Calulate Test error
centroid_V00 = calculate_gaussian_center(TOF_V00[np.newaxis,:], nbins = nbins, limits = 3) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[0]))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[2]))

Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1)
   
# Print MAE
MAE = np.mean(Error, axis = 1)
print(MAE[-1])

# Plot
plt.hist(test_dec0, bins = nbins, range = [12, 16], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, range = [12, 16], alpha = 0.5, label = 'Detector 1');
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
#--------------------------- BOOTSTRAPING ---------------------------------
# -------------------------------------------------------------------------

resolution_list = []
bias_list = []
MAE_list = []
centroid_00 = []
centroid_02 = []
centroid_20 = []

size_V00 = int(TOF_V00.shape[0]/10)
size_V02 = int(TOF_V02.shape[0]/10)
size_V20 = int(TOF_V20.shape[0]/10)

for i in range(10):
    centroid_V00 = calculate_gaussian_center(TOF_V00[None, i*size_V00 : (i+1)*size_V00], nbins = nbins, limits = 3) 
    params_V02, errors_V02 = get_gaussian_params(TOF_V02[i*size_V02 : (i+1)*size_V02], centroid_V00, range = 0.8, nbins = nbins)
    params_V00, errors_V00 = get_gaussian_params(TOF_V00[i*size_V00 : (i+1)*size_V00], centroid_V00, range = 0.8, nbins = nbins)
    params_V20, errors_V20 = get_gaussian_params(TOF_V20[i*size_V20 : (i+1)*size_V20], centroid_V00, range = 0.8, nbins = nbins)
    
    
    resolution = np.mean((params_V20[3], params_V00[3], params_V02[3]))
    resolution_list.append(resolution)
    
    centroids = np.array([params_V20[2], params_V00[2], params_V02[2]])
    bias = np.mean(abs(centroids - positions))
    bias_list.append(bias)

    error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[0]))
    error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
    error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[2]))

    Error = np.concatenate((error_V02, error_V00, error_V20), axis = 1)   
    MAE_list.append(np.mean(Error)) 

    centroid_00.append(params_V00[2])
    centroid_02.append(params_V02[2])
    centroid_20.append(params_V20[2])


print('Mean CTR: ', np.mean(np.array(resolution_list))*1000)
print('Std CTR: ', np.std(np.array(resolution_list))*1000)
print('Mean bias: ', np.mean(np.array(bias_list))*1000)
print('Std bias: ', np.std(np.array(bias_list))*1000)
print('Mean MAE: ', np.mean(np.array(MAE_list))*1000)
print('Std MAE: ', np.std(np.array(MAE_list))*1000)

print('Mean centroid 00: ', np.mean(np.array(centroid_00))*1000)
print('Std centroid 00: ', np.std(np.array(centroid_00))*1000)
print('Mean centroid 02: ', np.mean(np.array(centroid_02))*1000)
print('Std centroid 02: ', np.std(np.array(centroid_02))*1000)
print('Mean centroid 20: ', np.mean(np.array(centroid_20))*1000)
print('Std centroid 20: ', np.std(np.array(centroid_20))*1000)

