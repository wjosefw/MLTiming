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
moments_order = 7
start = 61
stop = 73
batch_size = 32
architecture = [moments_order, 3, 1, 1]    # KAN architecture
Num_Neurons = 16


# -------------------------------------------------------------------------
# ----------------------- MOVE TO REFERENCE -------------------------------
# -------------------------------------------------------------------------

mean_pulses_dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/'
mean_pulse_dec0 = np.load(os.path.join(mean_pulses_dir, 'reference_pulse_dec0.npz'))['data']
mean_pulse_dec1 = np.load(os.path.join(mean_pulses_dir, 'reference_pulse_dec1.npz'))['data']

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start, stop = stop, max_delay = int(stop-start), channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start, stop = stop, max_delay = int(stop-start), channel = 1)

TEST = np.stack((moved_pulses_test_dec0, moved_pulses_test_dec1), axis = 2)

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

 ## Calculate moments 
M_Test = momentos(test_data, order = moments_order)

params_dec0 = (np.array([0.46294019, 0.48189313, 0.46084979, 0.43525004, 0.4108511 ,
       0.3888638 , 0.36936704]), np.array([0.42321711, 0.37704465, 0.33838475, 0.30744539, 0.28236585,
       0.26169133, 0.24438544]))
params_dec1 = (np.array([0.34837473, 0.38529399, 0.37614394, 0.35929   , 0.34166016,
       0.32509272, 0.31004378]), np.array([0.4272395 , 0.35716153, 0.31630676, 0.286154  , 0.26244718,
       0.24318128, 0.22717464]))


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

#model_dec0_dir = os.path.join(dir, 'MLP_model_dec0')
#model_dec1_dir = os.path.join(dir, 'MLP_model_dec1')

#model_dec0_dir = os.path.join(dir, 'MLPWAVE_model_dec0')
#model_dec1_dir = os.path.join(dir, 'MLPWAVE_model_dec1')

model_dec0 = ConvolutionalModel(int(stop-start))
model_dec1 = ConvolutionalModel(int(stop-start))

#model_dec0 = KAN(architecture)
#model_dec1 = KAN(architecture)

#model_dec0 = MLP_Torch(NM = int(stop-start), NN = Num_Neurons, STD_INIT = 0.5)
#model_dec1 = MLP_Torch(NM = int(stop-start), NN = Num_Neurons, STD_INIT = 0.5)
         
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
centroid_V00 = calculate_gaussian_center(TOF_V00[np.newaxis,:], nbins = nbins, limits = 3) 

error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[2]))
error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[0]))

Error = np.concatenate((error_V02, error_V20, error_V00), axis = 1)
   
# Print MAE
MAE = np.mean(Error, axis = 1)
print(MAE[-1])


# Plot
plt.hist(test_dec0, bins = nbins, alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, alpha = 0.5, label = 'Detector 1');
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
    centroid_V00 = calculate_gaussian_center(TOF_V00[None, i*size_V00 : (i+1)*size_V00], nbins = nbins, limits = 3) 
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
    
