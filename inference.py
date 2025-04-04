import os 
import numpy as np
import torch
import argparse

# General settings
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Construct parser
parser = argparse.ArgumentParser()

parser.add_argument('--data', type = str, required = True, help = 'Path to data')
parser.add_argument('--model', type = str, required = True, help = 'Model to use (CNN, MLP, MLPWAVE OR KAN)')
args = parser.parse_args()

# Import functions    
from functions import ( get_mean_pulse_from_set, set_seed, momentos, 
                       normalize_given_params, move_to_reference,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel, MLP_Torch
from efficient_kan.src.efficient_kan import KAN

#Load data
test_data = np.load(args.data)['data'][:,:,0]
test_data = test_data[:,:,None]

print('Número de casos de test: ', test_data.shape[0])
set_seed(seed)   # Fix seeds

# -------------------------------------------------------------------------
# --------------------- IMPORTANT DEFINITIONS -----------------------------
# -------------------------------------------------------------------------

# Training parameters
Num_Neurons = 16
normalization_method = 'min-max'
moments_order = 3
architecture = [moments_order, 5, 1, 1]    # KAN architecture

# Data settings
time_step = 0.2  # Signal time step in ns
before = 8
after = 5
threshold = 0.1  # Reference threshold crop pulses

# -------------------------------------------------------------------------
# ----------------------- MOVE TO REFERENCE -------------------------------
# -------------------------------------------------------------------------

mean_pulse = get_mean_pulse_from_set(test_data, channel = 0)

# Get start and stop
crossing = calculate_slope_y_intercept(mean_pulse, time_step, threshold = threshold)

start = int(crossing/time_step) - before
stop = int(crossing/time_step) + after

delays_test, moved_pulses = move_to_reference(mean_pulse, test_data, start = start, stop = stop, channel = 0)

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

# Calculate moments 
M_Test = momentos(moved_pulses[:,:,None], order = moments_order)

params = (np.array([-0.07050748,  0.02451204,  0.04299015]), np.array([1.12753489, 0.93094554, 0.81081555]))

M_Test_norm = normalize_given_params(M_Test, params, channel = 0, method = normalization_method)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

if args.model == 'KAN':
    model_dir = os.path.join('Trained_Models/KAN_AG_model_dec0')
    model = KAN(architecture)

if args.model == 'MLP':
    model_dir = os.path.join('Trained_Models/MLP_AG_model_dec0')
    model = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)

if args.model == 'MLPWAVE':
    model_dir = os.path.join('Trained_Models/MLPWAVE_AG_model_dec0')
    model = MLP_Torch(NM = int(stop - start), NN = Num_Neurons, STD_INIT = 0.5)

if args.model == 'CNN':
    model_dir = os.path.join('Trained_Models/AG_model_dec0')
    model = ConvolutionalModel(int(stop - start))

model.load_state_dict(torch.load(model_dir, weights_only = True))
model.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

if args.model == 'CNN' or args.model == 'MLPWAVE':
    test = np.squeeze(model(torch.tensor(moved_pulses[:,None,:])).detach().numpy())

if args.model == 'KAN' or args.model == 'MLP':
    test = np.squeeze(model(torch.tensor(M_Test[:,:,0]).float()).detach().numpy())

decompressed_test = (test - time_step*delays_test)

# Save to txt as one column
decompressed_test_flat = decompressed_test.flatten()
np.savetxt('output_timing.txt', decompressed_test_flat, fmt = '%.6f')  



