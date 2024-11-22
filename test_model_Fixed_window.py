import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import ( calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, set_seed, momentos, normalize_given_params)
from Models import ConvolutionalModel
from efficient_kan.src.efficient_kan import KAN

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
start = 47
stop = 74
batch_size = 32
architecture = [5, 5, 1, 1]    # KAN architecture


# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

TEST = test_data[:,start:stop,:] 

# Calculate moments 
#MOMENTS_TEST = momentos(TEST, order = 5)
#
#params_dec0 = (np.array([0.45625919, 0.48014345, 0.46551475, 0.44413998, 0.42240204]), np.array([0.4199902 , 0.38190358, 0.34704695, 0.31804562, 0.29395941]))
#params_dec1 = (np.array([0.34293025, 0.38269777, 0.3791107 , 0.36606818, 0.35089854]), np.array([0.46452994, 0.37975432, 0.3360716 , 0.30489697, 0.28054114]))
#
#MOMENTS_TEST_norm_dec0 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
#MOMENTS_TEST_norm_dec1 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)
#MOMENTS_TEST = np.stack((MOMENTS_TEST_norm_dec0, MOMENTS_TEST_norm_dec1), axis = -1)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

dir = 'predictions/Convolutional/'
#dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/KAN_models'

model_dec0_dir = os.path.join(dir, 'Conv_model_dec0')
model_dec1_dir = os.path.join(dir, 'Conv_model_dec1')

#model_dec0_dir = os.path.join(dir, 'model_dec0')
#model_dec1_dir = os.path.join(dir, 'model_dec1')


model_dec0 = ConvolutionalModel(int(stop-start))
model_dec1 = ConvolutionalModel(int(stop-start))
#architecture = [5, 5, 1, 1]  
#model_dec0 = KAN(architecture)
#model_dec1 = KAN(architecture)

model_dec0.load_state_dict(torch.load(model_dec0_dir))
model_dec1.load_state_dict(torch.load(model_dec1_dir))
model_dec0.eval()
model_dec1.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

test_dec0 = np.squeeze(model_dec0(torch.tensor(TEST[:,None,:,0]).float()).detach().numpy())
test_dec1 = np.squeeze(model_dec1(torch.tensor(TEST[:,None,:,1]).float()).detach().numpy())


#test_dec0 = np.squeeze(model_dec0(torch.tensor(MOMENTS_TEST[:,:,0]).float()).detach().numpy())
#test_dec1 = np.squeeze(model_dec1(torch.tensor(MOMENTS_TEST[:,:,1]).float()).detach().numpy())

# Calculate TOF
TOF = test_dec0 - test_dec1

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
centroid_02_list = []
for i in range(1000):
    a_00 = np.random.choice(np.arange(0, TOF_V00.shape[0]), size = TOF_V00.shape[0], replace = True)
    a_02 = np.random.choice(np.arange(0, TOF_V02.shape[0]), size = TOF_V02.shape[0], replace = True)
    a_20 = np.random.choice(np.arange(0, TOF_V20.shape[0]), size = TOF_V20.shape[0], replace = True)

    centroid_V00 = calculate_gaussian_center(TOF_V00[None, a_00], nbins = nbins, limits = 3) 
    params_V02, errors_V02 = get_gaussian_params(TOF_V02[a_02], centroid_V00, range = 0.8, nbins = nbins)
    params_V00, errors_V00 = get_gaussian_params(TOF_V00[a_00], centroid_V00, range = 0.8, nbins = nbins)
    params_V20, errors_V20 = get_gaussian_params(TOF_V20[a_20], centroid_V00, range = 0.8, nbins = nbins)
    
    
    resolution = np.mean((params_V20[3], params_V00[3], params_V02[3]))
    resolution_list.append(resolution)
    
    centroid_02_list.append(params_V02[2])
    centroids = np.array([params_V20[2], params_V00[2], params_V02[2]])
    bias = np.mean(abs(centroids - positions))
    bias_list.append(bias)

    error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] - positions[2]))
    error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis] - positions[1]))
    error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - positions[0]))

    Error = np.concatenate((error_V02, error_V00, error_V20), axis = 1)   
    MAE_list.append(np.mean(Error)) 

print('Mean CTR: ', np.mean(np.array(resolution_list))*1000)
print('Std CTR: ', np.std(np.array(resolution_list))*1000)
print('Mean bias: ', np.mean(np.array(bias_list))*1000)
print('Std bias: ', np.std(np.array(bias_list))*1000)
print('Mean MAE: ', np.mean(np.array(MAE_list))*1000)
print('Std MAE: ', np.std(np.array(MAE_list))*1000)
print('02 std: ', np.std(np.array(centroid_02_list))*1000)

# -------------------------------------------------------------------------
#------------------------------- SHAP -------------------------------------
# -------------------------------------------------------------------------

import shap

model = model_dec0
channel = 0
waveforms = torch.tensor(TEST[:10,None,:,channel]).float()
# SHAP Explainer
explainer = shap.DeepExplainer(model, torch.tensor(TEST[-1000:,None,:,channel]).float())
shap_values = explainer.shap_values(waveforms, check_additivity = False)

shap_values_flat = shap_values[0][0].squeeze()
waveform_flat = waveforms.squeeze().detach().numpy()

# Plot waveform with SHAP values
for i in range(waveform_flat.shape[0]):
    plt.figure(figsize=(10, 6))
    plt.plot(waveform_flat[i,:], 'b-', label = "Waveform")
    plt.scatter(range(len(waveform_flat[i,:])), waveform_flat[i,:], c = shap_values_flat, cmap = "coolwarm", label = "SHAP values")
    plt.colorbar(label="Feature Importance")
    plt.title("SHAP Explanation for Waveform")
    plt.legend()
    plt.show()
    
# -------------------------------------------------------------------------
#-------------------------- INFERENCE TIME --------------------------------
# -------------------------------------------------------------------------

#import time
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#time_test = np.tile(MOMENTS_TEST[0,:,:] , (1000000, 1,1))
#model_dec0 = model_dec0.to(device)
#time_list_moments = []
#time_list_inference = []
#
## Start timer moments
#for i in range(10):
#    start_time_momentos = time.time()
#    M_time_test = momentos(time_test, order = moments_order)
#    end_time_momentos = time.time()
#    elapsed_time_momentos = end_time_momentos - start_time_momentos
#    time_list_moments.append(elapsed_time_momentos)
#time_array_moments = np.array(time_list_moments)
#
#
## Start timer inference
#for i in range(100):
#    start_time_inference= time.time()
#    with torch.no_grad():
#        assert not torch.is_grad_enabled()
#        output_time_test = model_dec0(torch.tensor(M_time_test[:,:,0]).float().to(device))
#    end_time_inference = time.time()
#    elapsed_time_inference = end_time_inference - start_time_inference
#    time_list_inference.append(elapsed_time_inference)
#time_array_inference = np.array(time_list_inference)
#
#
#print('Elapsed time momentos:', np.mean(time_array_moments), np.std(time_array_moments))
#print('Elapsed time inference:', np.mean(time_array_inference), np.std(time_array_inference))