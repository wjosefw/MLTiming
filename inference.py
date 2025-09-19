import os 
import numpy as np
import torch
import argparse
import yaml

# Import functions    
from functions import ( get_mean_pulse_from_set, set_seed, cfg_get, momentos, 
                       normalize_given_params, move_to_reference,
                       calculate_slope_y_intercept)
from Models import ConvolutionalModel, MLP_Torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Construct parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help = 'Path to YAML config')
parser.add_argument('--data', type = str, required = False, help = 'Path to data')
parser.add_argument('--model', type = str, required = False, help = 'Model to use (CNN, MLP, MLPWAVE OR KAN)')
args, extra = parser.parse_known_args() # parse known args first so the config can supply missing values

cfg = {}
if args.config:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

# Runtime options
seed = cfg_get(cfg, 'runtime.seed', 42)
set_seed(seed)

# fall back to config values when CLI flags are absent
data_path = args.data or cfg['data']['npz_path']
model_name = args.model or cfg['model']['name']

#Load data
test_data = np.load(data_path)['data']
test_data = test_data[:,:,None] # Some functions need the extra dimention

print('Number of events to infer:', test_data.shape[0])

# -------------------------------------------------------------------------
# --------------------- IMPORTANT DEFINITIONS -----------------------------
# -------------------------------------------------------------------------

output_filename = cfg_get(cfg, 'runtime.output_txt', '')
time_step = cfg_get(cfg, 'preprocessing.time_step_ns', 0.2)
threshold = cfg_get(cfg,'preprocessing.crossing_threshold', 0.1)
before = cfg_get(cfg,'preprocessing.crop.before_samples', 8)
after = cfg_get(cfg,'preprocessing.crop.after_samples', 5)

architecture = cfg_get(cfg,'preprocessing.model.architecture', [])
Num_Neurons = cfg_get(cfg,'preprocessing.model.num_neurons', 16)
moments_order = cfg_get(cfg,'preprocessing.model.moments_order', 3)

normalization_method = cfg_get(cfg,'preprocessing.normalization.method', 'min-max')
norm_params_min = cfg_get(cfg,'preprocessing.normalization.params_min', [])
norm_params_max = cfg_get(cfg,'preprocessing.normalization.params_max', [])

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
norm_params = (np.array(norm_params_min), np.array(norm_params_max))
M_Test_norm = normalize_given_params(M_Test, norm_params, channel = 0, method = normalization_method)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

if model_name == 'KAN':
    model_dir = os.path.join('Trained_Models/KAN_AG_model_dec0')
    model = KAN(architecture)

if model_name  == 'MLP':
    model_dir = os.path.join('Trained_Models/MLP_AG_model_dec0')
    model = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)

if model_name  == 'MLPWAVE':
    model_dir = os.path.join('Trained_Models/MLPWAVE_AG_model_dec0')
    model = MLP_Torch(NM = int(stop - start), NN = Num_Neurons, STD_INIT = 0.5)

if model_name  == 'CNN':
    model_dir = os.path.join('Trained_Models/AG_model_dec0')
    model = ConvolutionalModel(int(stop - start))

model.load_state_dict(torch.load(model_dir, weights_only = True))
model.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

if model_name  == 'CNN' or args.model == 'MLPWAVE':
    test = np.squeeze(model(torch.tensor(moved_pulses[:,None,:])).detach().numpy())

if model_name  == 'KAN' or args.model == 'MLP':
    test = np.squeeze(model(torch.tensor(M_Test[:,:,0]).float()).detach().numpy())

decompressed_test = (test - time_step*delays_test)

# Save to txt as one column
decompressed_test_flat = decompressed_test.flatten()
np.savetxt(output_filename, decompressed_test_flat, fmt = '%.6f')  



