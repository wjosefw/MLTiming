import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import Hyperparameters and Paths
from config_Gross_Adjustment import (
    device, Num_Neurons, before, after, normalization_method, moments_order, seed,
    architecture, time_step, nbins, Theoretical_TOF, positions, DATA_DIR, 
    REF_PULSE_SAVE_DIR, MODEL_SAVE_DIR, BASE_DIR
)

print(device)
sys.path.append(str(BASE_DIR.parent))

from functions import ( calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, set_seed, momentos, 
                       normalize_given_params, move_to_reference,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel, MLP_Torch
from efficient_kan.src.efficient_kan import KAN

#Load data
test_data_dict = {}
for i in range(np.min(positions), np.max(positions) + 1):  
    filename = f"Na22_norm_pos{i}_test.npz" if i >= 0 else f"Na22_norm_pos_min_{abs(i)}_test.npz"
    test_data_dict[i] = np.load(DATA_DIR / filename, mmap_mode="r")["data"]

test_data = np.concatenate((test_data_dict[-5], test_data_dict[-4], test_data_dict[-3], 
                            test_data_dict[-2], test_data_dict[-1], test_data_dict[0],  
                            test_data_dict[1], test_data_dict[2], test_data_dict[3],
                            test_data_dict[4], test_data_dict[5]), axis = 0)

print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)   # Fix seeds

# -------------------------------------------------------------------------
# ----------------------- MOVE TO REFERENCE -------------------------------
# -------------------------------------------------------------------------

mean_pulse_dec0 = np.load(os.path.join(REF_PULSE_SAVE_DIR, 'reference_pulse_dec0.npz'))['data']
mean_pulse_dec1 = np.load(os.path.join(REF_PULSE_SAVE_DIR, 'reference_pulse_dec1.npz'))['data']

# Get start and stop
crossing_dec0 = calculate_slope_y_intercept(mean_pulse_dec0, time_step, threshold = 0.1)
crossing_dec1 = calculate_slope_y_intercept(mean_pulse_dec1, time_step, threshold = 0.1)

start_dec0 = int(crossing_dec0/time_step) - before
start_dec1 = int(crossing_dec1/time_step) - before
stop_dec0 = int(crossing_dec0/time_step) + after
stop_dec1 = int(crossing_dec1/time_step) + after

delays_test_dec0, moved_pulses_test_dec0 = move_to_reference(mean_pulse_dec0, test_data, start = start_dec0, stop = stop_dec0, channel = 0)
delays_test_dec1, moved_pulses_test_dec1 = move_to_reference(mean_pulse_dec1, test_data, start = start_dec1, stop = stop_dec1, channel = 1)

TEST = np.stack((moved_pulses_test_dec0, moved_pulses_test_dec1), axis = 2)

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

# Calculate moments 
M_Test = momentos(TEST, order = moments_order)

params_dec0 = (np.array([-0.07050748,  0.02451204,  0.04299015]), np.array([1.12753489, 0.93094554, 0.81081555]))
params_dec1 = (np.array([-0.04730621,  0.01841856,  0.03066056]), np.array([1.01901136, 0.83459017, 0.72586663]))


M_Test_norm_dec0 = normalize_given_params(M_Test, params_dec0, channel = 0, method = normalization_method)
M_Test_norm_dec1 = normalize_given_params(M_Test, params_dec1, channel = 1, method = normalization_method)
M_Test = np.stack((M_Test_norm_dec0, M_Test_norm_dec1), axis = -1)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

#model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'KAN_AG_model_dec0')
#model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'KAN_AG_model_dec1')

#model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'MLP_AG_model_dec0')
#model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'MLP_AG_model_dec1')

#model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_AG_model_dec0')
#model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_AG_model_dec1')

model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec0')
model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'AG_model_dec1')

model_dec0 = ConvolutionalModel(int(stop_dec0-start_dec0))
model_dec1 = ConvolutionalModel(int(stop_dec1-start_dec1))

#model_dec0 = KAN(architecture)
#model_dec1 = KAN(architecture)

#model_dec0 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
#model_dec1 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)

#model_dec0 = MLP_Torch(NM = int(stop_dec0 - start_dec0), NN = Num_Neurons, STD_INIT = 0.5)
#model_dec1 = MLP_Torch(NM = int(stop_dec1 - start_dec1), NN = Num_Neurons, STD_INIT = 0.5)         

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

# Calculate TOF and decompress
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)

size = int(TOF.shape[0]/Theoretical_TOF.shape[0])
TOF_dict = {}  
for i in range(np.min(positions), np.max(positions) + 1):  
    TOF_dict[i] = TOF[(i + np.max(positions)) * size : (i + np.max(positions) + 1) * size]  # Assign slices dynamically

# Calulate Error
centroid_V00 = calculate_gaussian_center(TOF_dict[0][np.newaxis,:], nbins = nbins, limit = 6) 
error_dict = {} 
for i in range(np.min(positions), np.max(positions) + 1):    
    error_dict[i] = abs(TOF_dict[i] - centroid_V00 - Theoretical_TOF[i + np.max(positions)])  # Compute error per position

MAE = np.mean(list(error_dict.values()))   
print(MAE)

# Plot
plt.hist(test_dec0, bins = nbins, alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plt.figure(figsize = (16,6))
for i in range(np.min(positions), np.max(positions) + 1):    
    plot_gaussian(TOF_dict[i], centroid_V00, range = 0.6, label = 'pos' + str(i), nbins = nbins)
    params, errors = get_gaussian_params(TOF_dict[i], centroid_V00, range = 0.6, nbins = nbins)
    print(f"{i}: CENTROID(ns) = {params[1]:.4f} +/- {errors[2]:.5f}  FWHM(ns) = {params[2]:.4f} +/- {errors[3]:.5f}")

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

# -------------------------------------------------------------------------
#--------------------------- BOOTSTRAPING ---------------------------------
# -------------------------------------------------------------------------

num_subsets = 10 # Number of subsets for bootstraping

resolution = np.zeros((num_subsets,))
bias = np.zeros((num_subsets,))
MAE = np.zeros((num_subsets,))
size = int(size / num_subsets) # Divide the set in the subsets

for i in range(num_subsets):
    centroid_V00 = calculate_gaussian_center(TOF_dict[0][None, i*size : (i+1)*size], nbins = nbins) 

    resolution_dict = {} # Initialize dictionaries
    centroids_dict = {} 
    error_dict = {} 
    for j in range(np.min(positions), np.max(positions) + 1):    
        params, errors = get_gaussian_params(TOF_dict[j][i*size : (i+1)*size], centroid_V00, range = 0.6, nbins = nbins)
        error_dict[j] = abs(TOF_dict[j][i*size : (i+1)*size] - centroid_V00 - Theoretical_TOF[j + 5])  # Compute error per position
        resolution_dict[j] = params[2]
        centroids_dict[j] = params[1]
    
    centroids = list(centroids_dict.values())   
    bias[i] = np.mean(abs(centroids - Theoretical_TOF))  
    MAE[i] = np.mean(list(error_dict.values()))  
    resolution[i] = np.mean(list(resolution_dict.values()))  

print('Mean CTR: ', np.mean(resolution)*1000)
print('Std CTR: ', np.std(resolution)*1000)
print('Mean bias: ', np.mean(bias)*1000)
print('Std bias: ', np.std(bias)*1000)
print('Mean MAE: ', np.mean(MAE)*1000)
print('Std MAE: ', np.std(MAE)*1000)

# -------------------------------------------------------------------------
#-------------------------- INFERENCE TIME --------------------------------
# -------------------------------------------------------------------------

import time

time_test = np.tile(TEST[0,:,0] , (1000000, 1, 1))
model_dec0 = model_dec0.to(device)
time_list_move = []
time_list_moments = []
time_list_inference = []

# Start timer move to reference
move_time_test = np.tile(test_data[0,:,:] , (1000000, 1, 1))
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
        output_time_test = model_dec0(torch.tensor(TEST[:,None,:,0]).to(device))
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
    
