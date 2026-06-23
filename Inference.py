import os
import json
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

# Import functions
from functions import (set_seed, momentos, normalize_given_params,
                       extract_signal_window_by_fraction, select_channel,
                       calculate_gaussian_center, plot_gaussian, get_gaussian_params)
from Models import ConvolutionalModel, MLP_Torch
from efficient_kan.src.efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Construct parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, required = True, help = 'Path to data (.npy or .npz with key "data")')
parser.add_argument('--checkpoint', type = str, required = True, help = 'Path to a checkpoint saved by train_loop (its `<checkpoint>.json` metadata sidecar is loaded automatically)')
parser.add_argument('--checkpoint2', type = str, required = False, help = 'Path to a second checkpoint (detector 1). If given, both channels of --data are inferred and the coincidence time resolution is reported')
parser.add_argument('--channel', type = int, required = False, help = 'Detector channel to select if --data has shape (N, M, 2) and --checkpoint2 is not given')
parser.add_argument('--seed', type = int, default = 42, help = 'Random seed')
parser.add_argument('--output', type = str, default = 'output_timing.txt', help = 'Path to write the predicted timestamps (or TOF, in coincidence mode) to')
parser.add_argument('--nbins', type = int, default = 51, help = 'Number of histogram bins for the prediction/TOF histograms')
args = parser.parse_args()

set_seed(args.seed)

data_path = args.data
checkpoint_path = args.checkpoint
checkpoint_path2 = args.checkpoint2
channel = args.channel
output_filename = args.output
nbins = args.nbins

coincidence_mode = checkpoint_path2 is not None

# -------------------------------------------------------------------------
# ---------------------- LOAD CHECKPOINT METADATA --------------------------
# -------------------------------------------------------------------------

metadata_path = f'{checkpoint_path}.json'
if not os.path.isfile(metadata_path):
    raise FileNotFoundError(
        f"No metadata file found at '{metadata_path}'. Inference requires the "
        "'<checkpoint>.json' sidecar written by train_loop() alongside the checkpoint."
    )
with open(metadata_path) as f:
    metadata = json.load(f)

model_type = metadata['model_type']
architecture = metadata['architecture']
Num_Neurons = metadata['num_neurons']
moments_order = metadata['moments_order']
time_step = metadata['time_step_ns']
threshold = metadata['crossing_threshold']
before = metadata['before_samples']
after = metadata['after_samples']
normalization_method = metadata['normalization_method']
norm_params_min = metadata['normalization_params_min']
norm_params_max = metadata['normalization_params_max']

#Load data
test_data = np.load(data_path)

if test_data.ndim == 3:
    test_data_dec0 = test_data[:, :, 0] if coincidence_mode else select_channel(test_data, channel = channel)
else:
    test_data_dec0 = test_data

print('Number of events to infer:', test_data_dec0.shape[0])

# -------------------------------------------------------------------------
# ----------------------- MOVE TO REFERENCE --------------------------------
# -------------------------------------------------------------------------

moved_pulses, delays_test = extract_signal_window_by_fraction(test_data_dec0, time_step, fraction = threshold, window_low = before, window_high = after)

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

if model_type in ['KAN', 'MLP']:
    M_Test = momentos(moved_pulses[:, :, None], order = moments_order)
    norm_params = (np.array(norm_params_min), np.array(norm_params_max))
    M_Test_norm = normalize_given_params(M_Test, norm_params, channel = 0, method = normalization_method)

# -------------------------------------------------------------------------
#--------------------------- LOAD MODEL ----------------------------------
# -------------------------------------------------------------------------

if model_type == 'KAN':
    model = KAN(architecture)

elif model_type == 'MLP':
    model = MLP_Torch(NM = moments_order, NN = Num_Neurons, STD_INIT = 0.5)

elif model_type == 'MLPWAVE':
    model = MLP_Torch(NM = int(before + after), NN = Num_Neurons, STD_INIT = 0.5)

elif model_type == 'CNN':
    model = ConvolutionalModel(int(before + after))

else:
    raise ValueError(f"Unsupported model_type: {model_type}. This routine is for 'KAN', 'MLP', 'MLPWAVE' and 'CNN' models only.")

model.load_state_dict(torch.load(checkpoint_path, weights_only = True))
model.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

if model_type == 'CNN':
    test = np.squeeze(model(torch.tensor(moved_pulses[:, None, :])).detach().numpy())

if model_type == 'MLPWAVE':
    test = np.squeeze(model(torch.tensor(moved_pulses)).detach().numpy())

if model_type in ['KAN', 'MLP']:
    test = np.squeeze(model(torch.tensor(M_Test_norm).float()).detach().numpy())

decompressed_test = (test - time_step*delays_test)

if not coincidence_mode:
    plt.figure()
    plt.hist(decompressed_test, bins = nbins)
    plt.xlabel('Predicted time (ns)')
    plt.ylabel('Counts')
    plt.show()

    # Save to txt as one column
    decompressed_test_flat = decompressed_test.flatten()
    np.savetxt(output_filename, decompressed_test_flat, fmt = '%.6f')

# -------------------------------------------------------------------------
# ------------------- DETECTOR 1 (coincidence mode only) -------------------
# -------------------------------------------------------------------------

if coincidence_mode:
    test_data_dec1 = test_data[:, :, 1]

    metadata_path_dec1 = f'{checkpoint_path2}.json'
    if not os.path.isfile(metadata_path_dec1):
        raise FileNotFoundError(
            f"No metadata file found at '{metadata_path_dec1}'. Inference requires the "
            "'<checkpoint>.json' sidecar written by train_loop() alongside the checkpoint."
        )
    with open(metadata_path_dec1) as f:
        metadata_dec1 = json.load(f)

    model_type_dec1 = metadata_dec1['model_type']
    architecture_dec1 = metadata_dec1['architecture']
    Num_Neurons_dec1 = metadata_dec1['num_neurons']
    moments_order_dec1 = metadata_dec1['moments_order']
    time_step_dec1 = metadata_dec1['time_step_ns']
    threshold_dec1 = metadata_dec1['crossing_threshold']
    before_dec1 = metadata_dec1['before_samples']
    after_dec1 = metadata_dec1['after_samples']
    normalization_method_dec1 = metadata_dec1['normalization_method']
    norm_params_min_dec1 = metadata_dec1['normalization_params_min']
    norm_params_max_dec1 = metadata_dec1['normalization_params_max']

    moved_pulses_dec1, delays_test_dec1 = extract_signal_window_by_fraction(test_data_dec1, time_step_dec1, fraction = threshold_dec1, window_low = before_dec1, window_high = after_dec1)

    if model_type_dec1 in ['KAN', 'MLP']:
        M_Test_dec1 = momentos(moved_pulses_dec1[:, :, None], order = moments_order_dec1)
        norm_params_dec1 = (np.array(norm_params_min_dec1), np.array(norm_params_max_dec1))
        M_Test_norm_dec1 = normalize_given_params(M_Test_dec1, norm_params_dec1, channel = 0, method = normalization_method_dec1)

    if model_type_dec1 == 'KAN':
        model_dec1 = KAN(architecture_dec1)

    elif model_type_dec1 == 'MLP':
        model_dec1 = MLP_Torch(NM = moments_order_dec1, NN = Num_Neurons_dec1, STD_INIT = 0.5)

    elif model_type_dec1 == 'MLPWAVE':
        model_dec1 = MLP_Torch(NM = int(before_dec1 + after_dec1), NN = Num_Neurons_dec1, STD_INIT = 0.5)

    elif model_type_dec1 == 'CNN':
        model_dec1 = ConvolutionalModel(int(before_dec1 + after_dec1))

    else:
        raise ValueError(f"Unsupported model_type: {model_type_dec1}. This routine is for 'KAN', 'MLP', 'MLPWAVE' and 'CNN' models only.")

    model_dec1.load_state_dict(torch.load(checkpoint_path2, weights_only = True))
    model_dec1.eval()

    if model_type_dec1 == 'CNN':
        test_dec1 = np.squeeze(model_dec1(torch.tensor(moved_pulses_dec1[:, None, :])).detach().numpy())

    if model_type_dec1 == 'MLPWAVE':
        test_dec1 = np.squeeze(model_dec1(torch.tensor(moved_pulses_dec1)).detach().numpy())

    if model_type_dec1 in ['KAN', 'MLP']:
        test_dec1 = np.squeeze(model_dec1(torch.tensor(M_Test_norm_dec1).float()).detach().numpy())

    decompressed_test_dec1 = (test_dec1 - time_step_dec1*delays_test_dec1)

    # -------------------------------------------------------------------------
    # ------------------------ COINCIDENCE TIME RESOLUTION ---------------------
    # -------------------------------------------------------------------------

    TOF = decompressed_test - decompressed_test_dec1

    centroid = calculate_gaussian_center(TOF[np.newaxis, :], nbins = nbins, limit = 6)[0]
    (H, x0, FWHM), errors = get_gaussian_params(TOF, centroid, nbins = nbins, range = 0.6)

    print(f'Bias: {x0*1000:.2f} ps')
    print(f'CTR resolution (FWHM): {FWHM*1000:.2f} ps')

    plt.figure()
    plot_gaussian(TOF, centroid, nbins = nbins, range = 0.6, label = 'Coincidence TOF')
    plt.xlabel(r'$\Delta t$ (ns)')
    plt.ylabel('Counts')
    plt.show()

    np.savetxt(output_filename, TOF, fmt = '%.6f')
