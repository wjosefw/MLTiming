import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import ( calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, set_seed, momentos, 
                       normalize_given_params, move_to_reference)
from Models import ConvolutionalModel, MLP_Torch


#Load data
dir = '/home/josea/Pulsos15CM20250130/'
data0 = np.load(os.path.join(dir, 'Na22_norm_pos0.npz'))['data']
data1 = np.load(os.path.join(dir, 'Na22_norm_pos1.npz'))['data']
data2 = np.load(os.path.join(dir, 'Na22_norm_pos2.npz'))['data']
data3 = np.load(os.path.join(dir, 'Na22_norm_pos3.npz'))['data']
data4 = np.load(os.path.join(dir, 'Na22_norm_pos4.npz'))['data']
data5 = np.load(os.path.join(dir, 'Na22_norm_pos5.npz'))['data']
data6 = np.load(os.path.join(dir, 'Na22_norm_pos6.npz'))['data']

data_min_1 = np.load(os.path.join(dir, 'Na22_norm_pos_min_1.npz'))['data']
data_min_2 = np.load(os.path.join(dir, 'Na22_norm_pos_min_2.npz'))['data']
data_min_3 = np.load(os.path.join(dir, 'Na22_norm_pos_min_3.npz'))['data']
data_min_4 = np.load(os.path.join(dir, 'Na22_norm_pos_min_4.npz'))['data']
data_min_5 = np.load(os.path.join(dir, 'Na22_norm_pos_min_5.npz'))['data']
data_min_6 = np.load(os.path.join(dir, 'Na22_norm_pos_min_6.npz'))['data']

test_data = np.concatenate((data0[-10000:,:,:], 
                            data1[-10000:,:,:], data2[-10000:,:,:], data3[-10000:,:,:],
                            data4[-10000:,:,:], data5[-10000:,:,:], data6[-10000:,:,:],
                            data_min_1[-10000:,:,:], data_min_2[-10000:,:,:], data_min_3[-10000:,:,:],
                            data_min_4[-10000:,:,:], data_min_5[-10000:,:,:], data_min_6[-10000:,:,:]), axis = 0)

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
positions = 0.066*np.array([6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6])  # Expected time difference of each position
start_dec0 = 262
stop_dec0 = 275
start_dec1 = 262
stop_dec1 = 275
batch_size = 32
architecture = [moments_order, 3, 1, 1]    # KAN architecture
Num_Neurons = 32

# -------------------------------------------------------------------------
# ----------------------- MOVE TO REFERENCE -------------------------------
# -------------------------------------------------------------------------

mean_pulses_dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/'
mean_pulse_dec0 = np.load(os.path.join(mean_pulses_dir, 'reference_pulse_dec0.npz'))['data']
mean_pulse_dec1 = np.load(os.path.join(mean_pulses_dir, 'reference_pulse_dec1.npz'))['data']

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start_dec1, stop = stop_dec1, channel = 1)

TEST = np.stack((moved_pulses_test_dec0, moved_pulses_test_dec1), axis = 2)

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

# Calculate moments 
M_Test = momentos(TEST, order = moments_order)

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


TOF_0 = TOF[:10000] 
TOF_1 = TOF[ 10000:20000] 
TOF_2 = TOF[ 20000:30000] 
TOF_3 = TOF[ 30000:40000] 
TOF_4 = TOF[ 40000:50000] 
TOF_5 = TOF[ 50000:60000] 
TOF_6 = TOF[ 60000:70000] 
TOF_min_1 = TOF[70000:80000] 
TOF_min_2 = TOF[80000:90000] 
TOF_min_3 = TOF[90000:100000] 
TOF_min_4 = TOF[100000:110000] 
TOF_min_5 = TOF[110000:120000] 
TOF_min_6 = TOF[120000:] 

# Calulate Validation error
centroid_V00 = calculate_gaussian_center(TOF_0[np.newaxis,:], nbins = nbins, limit = 1) 

error_min_6 = abs((TOF_min_6 - centroid_V00[:, np.newaxis] - positions[0]))
error_min_5 = abs((TOF_min_5 - centroid_V00[:, np.newaxis] - positions[1]))
error_min_4 = abs((TOF_min_4 - centroid_V00[:, np.newaxis] - positions[2]))
error_min_3 = abs((TOF_min_3 - centroid_V00[:, np.newaxis] - positions[3]))
error_min_2 = abs((TOF_min_2 - centroid_V00[:, np.newaxis] - positions[4]))
error_min_1 = abs((TOF_min_1 - centroid_V00[:, np.newaxis] - positions[5]))
error_0 = abs((TOF_0 - centroid_V00[:, np.newaxis] - positions[6]))
error_1 = abs((TOF_1 - centroid_V00[:, np.newaxis] - positions[7]))
error_2 = abs((TOF_2 - centroid_V00[:, np.newaxis] - positions[8]))
error_3 = abs((TOF_3 - centroid_V00[:, np.newaxis] - positions[9]))
error_4 = abs((TOF_4 - centroid_V00[:, np.newaxis] - positions[10]))
error_5 = abs((TOF_5 - centroid_V00[:, np.newaxis] - positions[11]))
error_6 = abs((TOF_6 - centroid_V00[:, np.newaxis] - positions[12]))

# Get MAE
Error = np.concatenate((error_0, 
                        error_1, error_2, error_3,
                        error_4, error_5, error_6, 
                        error_min_1, error_min_2, error_min_3,
                        error_min_4, error_min_5, error_min_6), axis = 1)   

# Print MAE
MAE = np.mean(Error, axis = 1)
print(MAE[-1])


# Plot
plt.hist(test_dec0, bins = nbins, alpha = 0.5, range = [0.0,0.8], label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, alpha = 0.5, range = [0.0,0.8], label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plot_gaussian(TOF_0, centroid_V00, range = 0.6, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_3, centroid_V00, range = 0.6, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_5, centroid_V00, range = 0.6, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_min_3, centroid_V00, range = 0.6, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_min_5, centroid_V00, range = 0.6, label = '-0.2 ns offset', nbins = nbins)

params_0, errors_0 = get_gaussian_params(TOF_0, centroid_V00, range = 0.6, nbins = nbins)
params_1, errors_1 = get_gaussian_params(TOF_1, centroid_V00, range = 0.6, nbins = nbins)
params_2, errors_2 = get_gaussian_params(TOF_2, centroid_V00, range = 0.6, nbins = nbins)
params_3, errors_3 = get_gaussian_params(TOF_3, centroid_V00, range = 0.6, nbins = nbins)
params_4, errors_4 = get_gaussian_params(TOF_4, centroid_V00, range = 0.6, nbins = nbins)
params_5, errors_5 = get_gaussian_params(TOF_5, centroid_V00, range = 0.6, nbins = nbins)
params_6, errors_6 = get_gaussian_params(TOF_6, centroid_V00, range = 0.6, nbins = nbins)
params_min_1, errors_min_1 = get_gaussian_params(TOF_min_1, centroid_V00, range = 0.6, nbins = nbins)
params_min_2, errors_min_2 = get_gaussian_params(TOF_min_2, centroid_V00, range = 0.6, nbins = nbins)
params_min_3, errors_min_3 = get_gaussian_params(TOF_min_3, centroid_V00, range = 0.6, nbins = nbins)
params_min_4, errors_min_4 = get_gaussian_params(TOF_min_4, centroid_V00, range = 0.6, nbins = nbins)
params_min_5, errors_min_5 = get_gaussian_params(TOF_min_5, centroid_V00, range = 0.6, nbins = nbins)
params_min_6, errors_min_6 = get_gaussian_params(TOF_min_6, centroid_V00, range = 0.6, nbins = nbins)

print("min 6: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_6[1], errors_min_6[1], params_min_6[2], errors_min_6[2]))
print("min 5: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_5[1], errors_min_5[1], params_min_5[2], errors_min_5[2]))
print("min 4: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_4[1], errors_min_4[1], params_min_4[2], errors_min_4[2]))
print("min 3: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_3[1], errors_min_3[1], params_min_3[2], errors_min_3[2]))
print("min 2: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_2[1], errors_min_2[1], params_min_2[2], errors_min_2[2]))
print("min 1: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_min_1[1], errors_min_1[1], params_min_1[2], errors_min_1[2]))
print("0: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_0[1], errors_0[1], params_0[2], errors_0[2]))
print("1: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_1[1], errors_1[1], params_1[2], errors_1[2]))
print("2: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_2[1], errors_2[1], params_2[2], errors_2[2]))
print("3: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_3[1], errors_3[1], params_3[2], errors_3[2]))
print("4: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_4[1], errors_4[1], params_4[2], errors_4[2]))
print("5: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_5[1], errors_5[1], params_5[2], errors_5[2]))
print("6: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_6[1], errors_6[1], params_6[2], errors_6[2]))

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

for i in range(10):
    centroid_V00 = calculate_gaussian_center(TOF_0[None, i*1000 : (i+1)*1000], nbins = nbins) 
    params_0, errors_0 = get_gaussian_params(TOF_0[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_1, errors_1 = get_gaussian_params(TOF_1[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_2, errors_2 = get_gaussian_params(TOF_2[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_3, errors_3 = get_gaussian_params(TOF_3[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_4, errors_4 = get_gaussian_params(TOF_4[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_5, errors_5 = get_gaussian_params(TOF_5[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_6, errors_6 = get_gaussian_params(TOF_6[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_min_1, errors_min_1 = get_gaussian_params(TOF_min_1[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_min_2, errors_min_2 = get_gaussian_params(TOF_min_2[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_min_3, errors_min_3 = get_gaussian_params(TOF_min_3[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_min_4, errors_min_4 = get_gaussian_params(TOF_min_4[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_min_5, errors_min_5 = get_gaussian_params(TOF_min_5[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)
    params_min_6, errors_min_6 = get_gaussian_params(TOF_min_6[i*1000 : (i+1)*1000], centroid_V00, range = 0.6, nbins = nbins)

    resolution = np.mean([params_min_6[2],  params_min_5[2], params_min_4[2],
                          params_min_3[2],  params_min_2[2], params_min_1[2],
                          params_0[2],  
                          params_1[2],  params_2[2], params_3[2], 
                          params_4[2],  params_5[2], params_6[2],  
                          ])
    
    resolution_list.append(resolution)
    
    centroids = np.array([params_min_6[1],  params_min_5[1], params_min_4[1],
                         params_min_3[1],  params_min_2[1], params_min_1[1],
                         params_0[1],  
                         params_1[1],  params_2[1], params_3[1], 
                         params_4[1],  params_5[1], params_6[1]])
    bias = np.mean(abs(centroids - positions))
    bias_list.append(bias)

    error_min_6 = abs((TOF_min_6 - centroid_V00[:, np.newaxis] - positions[0]))
    error_min_5 = abs((TOF_min_5 - centroid_V00[:, np.newaxis] - positions[1]))
    error_min_4 = abs((TOF_min_4 - centroid_V00[:, np.newaxis] - positions[2]))
    error_min_3 = abs((TOF_min_3 - centroid_V00[:, np.newaxis] - positions[3]))
    error_min_2 = abs((TOF_min_2 - centroid_V00[:, np.newaxis] - positions[4]))
    error_min_1 = abs((TOF_min_1 - centroid_V00[:, np.newaxis] - positions[5]))
    error_0 = abs((TOF_0 - centroid_V00[:, np.newaxis] - positions[6]))
    error_1 = abs((TOF_1 - centroid_V00[:, np.newaxis] - positions[7]))
    error_2 = abs((TOF_2 - centroid_V00[:, np.newaxis] - positions[8]))
    error_3 = abs((TOF_3 - centroid_V00[:, np.newaxis] - positions[9]))
    error_4 = abs((TOF_4 - centroid_V00[:, np.newaxis] - positions[10]))
    error_5 = abs((TOF_5 - centroid_V00[:, np.newaxis] - positions[11]))
    error_6 = abs((TOF_6 - centroid_V00[:, np.newaxis] - positions[12]))
    
    Error = np.concatenate((error_0, 
                        error_1, error_2, error_3,
                        error_4, error_5, error_6, 
                        error_min_1, error_min_2, error_min_3,
                        error_min_4, error_min_5, error_min_6), axis = 1)   

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

time_test = np.tile(TEST[0,:,0] , (1000000, 1, 1))
model_dec0 = model_dec0.to(device)
time_list_move = []
time_list_moments = []
time_list_inference = []

# Start timer move to reference
move_time_test = np.tile(test_data[0,:,:] , (5000, 1, 1))
for i in range(10):
    start_time_move = time.time()
    delays, moved_pulses = move_to_reference(mean_pulse_dec0, move_time_test, start = start_dec0, stop = stop_dec0, channel = 0)
    end_time_move = time.time()
    elapsed_time_move = end_time_move - start_time_move
    time_list_move.append(elapsed_time_move)
time_array_move = np.array(time_list_move)

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

time_array_move = time_array_move[1:]  
time_array_moments = time_array_moments[1:]   # We discard the first loop because it is artificially slower due to 'warm-up' effect 
time_array_inference = time_array_inference[1:]

print('Elapsed time move:', np.mean(time_array_move), np.std(time_array_move))
print('Elapsed time momentos:', np.mean(time_array_moments), np.std(time_array_moments))
print('Elapsed time inference:', np.mean(time_array_inference), np.std(time_array_inference))
print('Elapsed time momentos + inference :', np.mean(time_array_moments) + np.mean(time_array_inference), np.std(time_array_moments) + np.std(time_array_inference))


# -------------------------------------------------------------------------
#------------------------------- SHAP -------------------------------------
# -------------------------------------------------------------------------

import shap

model = model_dec0
channel = 0
waveforms = mean_pulse_dec0[start_dec0:stop_dec0]
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
    
