import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import Hyperparameters and Paths
from config_Fixed_Window import (device, delay_time, time_step, nbins, 
                                 seed, epochs, lr, batch_size, save, 
                                 positions, start, stop, Theoretical_TOF,
                                 BASE_DIR, MODEL_SAVE_DIR, DATA_DIR, step_size,
                                 moments_order, normalization_method)

print(device)
sys.path.append(str(BASE_DIR.parent))

from efficient_kan.src.efficient_kan import KAN

from functions import ( calculate_gaussian_center, plot_gaussian, 
                       get_gaussian_params, set_seed, momentos, normalize_given_params)
from Models import ConvolutionalModel, MLP_Torch
from Dataset import Datos_LAB_GFN

#Load data
dataset = Datos_LAB_GFN(data_dir = DATA_DIR, positions = positions, step_size = step_size)
test_data = dataset.load_data()

print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)   # Fix seeds

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

TEST = test_data[:,start:stop,:] 

## Calculate moments 
M_Test = momentos(TEST, order = moments_order)

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

model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'model_dec0')
model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'model_dec1')

#model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'Conv_model_dec0')
#model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'Conv_model_dec1')

#model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'MLP_model_dec0')
#model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'MLP_model_dec1')

#model_dec0_dir = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_model_dec0')
#model_dec1_dir = os.path.join(MODEL_SAVE_DIR, 'MLPWAVE_model_dec1')

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

# Calculate TOF and decompress
TOF = test_dec0 - test_dec1

size = int(TOF.shape[0]/Theoretical_TOF.shape[0]) # Size of slice
TOF_dict = dataset.get_TOF_slices_eval(TOF, size)

# Calulate Error
centroid_V00 = calculate_gaussian_center(TOF_dict[0][np.newaxis,:], nbins = nbins, limit = 6) 

error_dict = dataset.compute_error(centroid_V00) # Get error of each position
MAE = np.mean(list(error_dict.values()))   
print(MAE)

# Plot
plt.hist(test_dec0, bins = nbins, range = [18, 22], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, range = [18, 22], alpha = 0.5, label = 'Detector 1');
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