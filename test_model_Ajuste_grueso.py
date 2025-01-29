import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import ( calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, set_seed, momentos, normalize_given_params,
                       move_to_reference, create_dataloaders)
from Models import ConvolutionalModel, MLP_Torch


#Load data
dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
test_data_82 = np.load(os.path.join(dir, 'Na22_82_norm_ALBA_test.npz'))['data']
test_data_55 = np.load(os.path.join(dir, 'Na22_55_norm_ALBA_test.npz'))['data']
test_data_28 = np.load(os.path.join(dir, 'Na22_28_norm_ALBA_test.npz'))['data']

test_data  = np.concatenate((test_data_55, test_data_28, test_data_82), axis = 0)
print('Número de casos de test: ', test_data.shape[0])


# -------------------------------------------------------------------------
# ---------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

set_seed(42)                               # Fix seeds
nbins = 71                                 # Num bins for all histograms
time_step = 0.2                            # Signal time step in ns
positions = np.array([0.2, 0.0, -0.2])
normalization_method = 'standardization'
moments_order = 6
start_dec0 = 60
stop_dec0 = 74
start_dec1 = 61
stop_dec1 = 75
batch_size = 32
architecture = [moments_order, 3, 1, 1]    # KAN architecture
Num_Neurons = 32

def move_to_reference(reference, pulse_set, start=50, stop=80, max_delay=10, channel=0):

    if (stop - start) < max_delay:
        raise ValueError("Window size (stop-start) cannot be smaller than max_delay")

    # Extract the reference window
    reference_pulse = reference[start:stop]
    delays = []
    aligned_pulses = []

    for i in range(pulse_set.shape[0]):
        mse = []
        segments = []
        
        # Extract the current pulse channel and sliding window
        pulse = pulse_set[i, :, channel]
        for i in range(0, pulse_set.shape[1] - int(stop - start)):
            start_idx = i
            stop_idx = i + int(stop - start)

            # Ensure valid indices within bounds of the pulse array
            if start_idx < 0 or stop_idx > len(pulse):
                continue

            # Extract the sliding window segment
            segment = pulse[start_idx:stop_idx]
            mse.append(np.mean((reference_pulse - segment) ** 2))
            segments.append(segment)

        # Find the shift with the minimal MSE
        mse = np.array(mse)
        min_mse_index = np.argmin(mse)
        optimal_start = range(0, pulse_set.shape[1] - int(stop - start))[min_mse_index]
        optimal_shift = start - optimal_start  
        
        delays.append(optimal_shift)
        aligned_pulses.append(segments[min_mse_index])
        

    return np.array(delays), np.array(aligned_pulses)

# -------------------------------------------------------------------------
# ----------------------- MOVE TO REFERENCE -------------------------------
# -------------------------------------------------------------------------

mean_pulses_dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/'
mean_pulse_dec0 = np.load(os.path.join(mean_pulses_dir, 'reference_pulse_dec0.npz'))['data']
mean_pulse_dec1 = np.load(os.path.join(mean_pulses_dir, 'reference_pulse_dec1.npz'))['data']

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start_dec0, stop = stop_dec0, max_delay = int(stop_dec0-start_dec0), channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start_dec1, stop = stop_dec1, max_delay = int(stop_dec1-start_dec1), channel = 1)

print(type(moved_pulses_test_dec0))
TEST = np.stack((moved_pulses_test_dec0, moved_pulses_test_dec1), axis = 2)

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

# Calculate moments 
M_Test = momentos(TEST, order = moments_order)

#params_dec0 = (np.array([0.38743577, 0.35521059, 0.32091923, 0.29320879, 0.27105946,
#       0.25315543, 0.23847421, 0.22627119]), np.array([0.18359224, 0.15205655, 0.13110244, 0.11621708, 0.10512331,
#       0.09655463, 0.08975309, 0.08423733]))
#params_dec1 = (np.array([0.28284467, 0.27656352, 0.25491724, 0.23544002, 0.21921152,
#       0.20579795, 0.19464108, 0.18527334]), np.array([0.15871774, 0.1323155 , 0.11494621, 0.10255006, 0.09325127,
#       0.08601955, 0.08024191, 0.07552924]))
#
params_dec0 = (np.array([0.37360714, 0.34350668, 0.310697  , 0.2840613 , 0.26272417,
       0.24545365]), np.array([0.19064454, 0.15812544, 0.13650124, 0.12110903, 0.10962155,
       0.10073992]))

params_dec1 = (np.array([0.2666009 , 0.26276701, 0.24283218, 0.22459564, 0.20930532,
       0.19662505]), np.array([0.16339829, 0.13645537, 0.11867063, 0.10595505, 0.09640735,
       0.08897757]))

M_Test_norm_dec0 = normalize_given_params(M_Test, params_dec0, channel = 0, method = normalization_method)
M_Test_norm_dec1 = normalize_given_params(M_Test, params_dec1, channel = 1, method = normalization_method)
M_Test = np.stack((M_Test_norm_dec0, M_Test_norm_dec1), axis = -1)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

dir = 'predictions/Convolutional/'
#dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/KAN_models'

model_dec0_dir = os.path.join(dir, 'AG_model_dec0')
model_dec1_dir = os.path.join(dir, 'AG_model_dec1')

#model_dec0_dir = os.path.join(dir, 'Conv_model_dec0')
#model_dec1_dir = os.path.join(dir, 'Conv_model_dec1')

#model_dec0_dir = os.path.join(dir, 'MLP_AG_model_dec0')
#model_dec1_dir = os.path.join(dir, 'MLP_AG_model_dec1')

#model_dec0_dir = os.path.join(dir, 'MLPWAVE_model_dec0')
#model_dec1_dir = os.path.join(dir, 'MLPWAVE_model_dec1')

model_dec0 = ConvolutionalModel(int(stop_dec0-start_dec0))
model_dec1 = ConvolutionalModel(int(stop_dec1-start_dec1))

#model_dec0 = KAN(architecture)
#model_dec1 = KAN(architecture)

#model_dec0 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
#model_dec1 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
         
model_dec0.load_state_dict(torch.load(model_dec0_dir))
model_dec1.load_state_dict(torch.load(model_dec1_dir))
model_dec0.eval()
model_dec1.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

test_dec0 = np.squeeze(model_dec0(torch.tensor(TEST[:,None,:,0])).detach().numpy())
test_dec1 = np.squeeze(model_dec1(torch.tensor(TEST[:,None,:,1])).detach().numpy())

#test_dec0 = np.squeeze(model_dec0(torch.tensor(M_Test[:,:,0]).float()).detach().numpy())
#test_dec1 = np.squeeze(model_dec1(torch.tensor(M_Test[:,:,1]).float()).detach().numpy())

# Calculate TOF
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)

TOF_V00 = TOF[:test_data_55.shape[0]] 
TOF_V02 = TOF[test_data_55.shape[0] : test_data_55.shape[0] + test_data_28.shape[0]] 
TOF_V20 = TOF[test_data_55.shape[0] + test_data_28.shape[0]:] 


# Calulate Test error
centroid_V00 = calculate_gaussian_center(TOF_V00[np.newaxis,:], nbins = nbins) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[2]))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[0]))

Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1)
   
# Print MAE
MAE = np.mean(Error, axis = 1)
print(MAE[-1])


# Plot
plt.hist(test_dec0, bins = nbins, alpha = 0.5, range = [0.2,0.8], label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, alpha = 0.5, range = [0.2,0.8], label = 'Detector 1');
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
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
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
    centroid_V00 = calculate_gaussian_center(TOF_V00[None, i*size_V00 : (i+1)*size_V00], nbins = nbins) 
    params_V02, errors_V02 = get_gaussian_params(TOF_V02[i*size_V02 : (i+1)*size_V02], centroid_V00, range = 0.8, nbins = nbins)
    params_V00, errors_V00 = get_gaussian_params(TOF_V00[i*size_V00 : (i+1)*size_V00], centroid_V00, range = 0.8, nbins = nbins)
    params_V20, errors_V20 = get_gaussian_params(TOF_V20[i*size_V20 : (i+1)*size_V20], centroid_V00, range = 0.8, nbins = nbins)
    
    
    resolution = np.mean((params_V20[3], params_V00[3], params_V02[3]))
    resolution_list.append(resolution)
    
    centroids = np.array([params_V20[2], params_V00[2], params_V02[2]])
    bias = np.mean(abs(centroids - positions))
    bias_list.append(bias)

    error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[2]))
    error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
    error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[0]))

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


# -------------------------------------------------------------------------
#-------------------------- INFERENCE TIME --------------------------------
# -------------------------------------------------------------------------

import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

time_test = np.tile(TEST[0,:,0] , (1000000, 1, 1))
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
for i in range(10):
    start_time_inference = time.time()
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        #output_time_test = model_dec0(torch.tensor(M_time_test[:,:,:]).float().to(device))
        output_time_test = np.squeeze(model_dec0(torch.tensor(TEST[:,None,:,0]).to(device)).cpu().detach().numpy())
    end_time_inference = time.time()
    elapsed_time_inference = end_time_inference - start_time_inference
    time_list_inference.append(elapsed_time_inference)
time_array_inference = np.array(time_list_inference)


time_array_moments = time_array_moments[1:]   # We discard the first loop because it is artificially slower due to 'warm-up' effect 
time_array_inference = time_array_inference[1:]

print('Elapsed time momentos:', np.mean(time_array_moments), np.std(time_array_moments))
print('Elapsed time inference:', np.mean(time_array_inference), np.std(time_array_inference))
print('Elapsed time momentos + inference :', np.mean(time_array_moments) + np.mean(time_array_inference), np.std(time_array_moments) + np.std(time_array_inference))


# -------------------------------------------------------------------------
#------------------------------- SHAP -------------------------------------
# -------------------------------------------------------------------------

import shap

model = model_dec0
channel = 0
waveforms = np.mean(TEST[:, :, channel], axis = 0)
waveforms = torch.tensor(waveforms[None, None, :]).float()

# SHAP Explainer
explainer = shap.DeepExplainer(model, torch.tensor(TEST[:, None, :, channel]).to(device))
shap_values = explainer.shap_values(waveforms, check_additivity = False)

shap_values_flat = shap_values[0][0].squeeze()
waveform_flat = waveforms.squeeze().detach().numpy()

# Plot waveform with SHAP values
plt.figure(figsize=(10, 6))
plt.plot(waveform_flat, 'b-', label = "Waveform")
plt.scatter(range(len(waveform_flat)), waveform_flat, c = shap_values_flat, cmap = "coolwarm", label = "SHAP values")
plt.colorbar(label = "Feature Importance")
plt.title("SHAP Explanation for Waveform")
plt.legend()
plt.show()
    
