import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

# Import Hyperparameters and Paths
from config import (
    device, Num_Neurons, before, after, normalization_method, moments_order, seed,
    architecture, nbins, threshold, DATA_DIR, REF_PULSE_SAVE_DIR, 
    MODEL_SAVE_DIR, BASE_DIR, before, after)

print(device)
sys.path.append(str(BASE_DIR.parent))

from functions import ( calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, set_seed, momentos, 
                       normalize_given_params, extract_signal_window_by_fraction)
from Models import ConvolutionalModel, MLP_Torch
from Dataset import Datos_LAB_GFN
from efficient_kan.src.efficient_kan import KAN

# Construct parser
parser = argparse.ArgumentParser()

parser.add_argument('--model', type = str, required = True, help = 'Model to use (CNN, MLP, MLPWAVE OR KAN)')
args = parser.parse_args()

#Load data
dataset = Datos_LAB_GFN(data_dir = DATA_DIR)
test_data = dataset.load_test_data()
time_step, positions, Theoretical_TOF = dataset.load_params() # Load data parameters

print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)   # Fix seeds

# -------------------------------------------------------------------------
# ----------------------- MOVE TO REFERENCE -------------------------------
# -------------------------------------------------------------------------

# Delays
moved_pulses_test_dec0, delays_test_dec0  = extract_signal_window_by_fraction(test_data[:,:,0], time_step, fraction = threshold, window_low = before, window_high = after)
moved_pulses_test_dec1, delays_test_dec1  = extract_signal_window_by_fraction(test_data[:,:,1], time_step, fraction = threshold, window_low = before, window_high = after)

TEST = np.stack((moved_pulses_test_dec0, moved_pulses_test_dec1), axis = 2)

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

# Calculate moments 
M_Test = momentos(TEST, order = moments_order)

params_dec0 = (np.array([-0.07050748,  0.02451204,  0.04299015]), np.array([1.12753489, 0.93094554, 0.81081555]))
params_dec1 = (np.array([-0.04730621,  0.01841856,  0.03066056]), np.array([1.01901136, 0.83459017, 0.72586663]))

#params_dec0 = (np.array([-0.07050748,  0.02451204,  0.04436237,  0.04975697]), np.array([1.17041325, 0.99355343, 0.86337464, 0.76888418]))
#params_dec1 = (np.array([-0.00929751,  0.04495382,  0.0527845 ,  0.05432888]), np.array([1.23438813, 1.05928715, 0.92953213, 0.83420514]))



M_Test_norm_dec0 = normalize_given_params(M_Test, params_dec0, channel = 0, method = normalization_method)
M_Test_norm_dec1 = normalize_given_params(M_Test, params_dec1, channel = 1, method = normalization_method)
M_Test = np.stack((M_Test_norm_dec0, M_Test_norm_dec1), axis = -1)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------


if args.model == 'KAN':
    model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'KAN_model_dec0')
    model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'KAN_model_dec1')
    model_dec0 = KAN(architecture)
    model_dec1 = KAN(architecture)

elif args.model == 'MLP':
    model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'MLP_model_dec0')
    model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'MLP_model_dec1')
    model_dec0 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)
    model_dec1 = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)

elif args.model == 'MLPWAVE':
    model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_model_dec0')
    model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_model_dec1')
    model_dec0 = MLP_Torch(NM = int(before + after), NN = Num_Neurons, STD_INIT = 0.5)
    model_dec1 = MLP_Torch(NM = int(before + after), NN = Num_Neurons, STD_INIT = 0.5)

elif args.model == 'CNN':
    model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'CNN_model_dec0')
    model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'CNN_model_dec1')
    model_dec0 = ConvolutionalModel(int(before + after))
    model_dec1 = ConvolutionalModel(int(before + after))

model_dec0.load_state_dict(torch.load(model_dec0_dir, weights_only = True))
model_dec1.load_state_dict(torch.load(model_dec1_dir, weights_only = True))
model_dec0.eval()
model_dec1.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

if args.model in ['CNN', 'MLPWAVE']:
    test_dec0 = np.squeeze(model_dec0(torch.tensor(TEST[:,None,:,0])).detach().numpy())
    test_dec1 = np.squeeze(model_dec1(torch.tensor(TEST[:,None,:,1])).detach().numpy())

elif args.model in ['KAN', 'MLP']:
    test_dec0 = np.squeeze(model_dec0(torch.tensor(M_Test[:,:,0]).float()).detach().numpy())
    test_dec1 = np.squeeze(model_dec1(torch.tensor(M_Test[:,:,1]).float()).detach().numpy())

# Calculate TOF and decompress
TOF = (test_dec0 - time_step*delays_test_dec0) - (test_dec1 - time_step*delays_test_dec1)
TOF_dict = dataset.get_TOF_slices_eval(TOF)

# Calulate Error
centroid_V00 = calculate_gaussian_center(TOF_dict[0][np.newaxis,:], nbins = nbins, limit = 6) 

error_dict = dataset.compute_error(centroid_V00) # Get error of each position
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
plt.xlabel(r'$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()

# -------------------------------------------------------------------------
#--------------------------- BOOTSTRAPING ---------------------------------
# -------------------------------------------------------------------------

num_subsets = 10 # Number of subsets for bootstraping
size = int(TOF.shape[0]/Theoretical_TOF.shape[0]) # Size of slice
subset_size = int(size / num_subsets) # Divide the set in the subsets

resolution = np.zeros((num_subsets,))
bias = np.zeros((num_subsets,))
MAE = np.zeros((num_subsets,))

for i in range(num_subsets):
    subset_slice = slice(i * subset_size, (i + 1) * subset_size)
    centroid_V00 = calculate_gaussian_center(TOF_dict[0][None, subset_slice], nbins = nbins) 

    resolution_dict = {} # Initialize dictionaries
    centroids_dict = {} 
    error_dict = {} 
    for j in range(np.min(positions), np.max(positions) + 1):    
        params, errors = get_gaussian_params(TOF_dict[j][subset_slice], centroid_V00, range = 0.6, nbins = nbins)
        error_dict[j] = abs(TOF_dict[j][subset_slice] - centroid_V00 - Theoretical_TOF[j + np.max(positions)])  # Compute error per position
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

