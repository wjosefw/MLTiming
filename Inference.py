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
from Models import build_model

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

# -------------------------------------------------------------------------
# ---------------------------- PREDICT ------------------------------------
# -------------------------------------------------------------------------

def predict(checkpoint_path, pulses):
    """Timestamps (ns from the first sample) of pulses (N, M) by a checkpoint saved by train_loop:
    its '<checkpoint>.json' metadata gives the model and the preprocessing (crop, moments)."""
    metadata_path = f'{checkpoint_path}.json'
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"No metadata file found at '{metadata_path}'. Inference requires the "
            "'<checkpoint>.json' sidecar written by train_loop() alongside the checkpoint."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    time_step = metadata['time_step_ns']
    crops, delays = extract_signal_window_by_fraction(pulses, time_step, fraction = metadata['crossing_threshold'],
                                                      window_low = metadata['before_samples'], window_high = metadata['after_samples'])

    model = build_model(metadata)
    model.load_state_dict(torch.load(checkpoint_path, weights_only = True))
    model.eval()

    with torch.no_grad():
        if metadata['model_type'] == 'CNN':
            pred = model(torch.tensor(crops[:, None, :]))
        elif metadata['model_type'] == 'MLPWAVE':
            pred = model(torch.tensor(crops))
        else:   # MLP, KAN: on the normalised moments of the crop
            moments = momentos(crops[:, :, None], order = metadata['moments_order'])
            params = (np.array(metadata['normalization_params_min']), np.array(metadata['normalization_params_max']))
            moments = normalize_given_params(moments, params, channel = 0, method = metadata['normalization_method'])
            pred = model(torch.tensor(moments).float())

    return np.squeeze(pred.numpy()) - time_step * delays

# -------------------------------------------------------------------------
# ------------------------------ RESULTS ----------------------------------
# -------------------------------------------------------------------------

data = np.load(args.data)
if args.data.endswith('.npz'):
    data = data['data']

if args.checkpoint2 is None:
    pulses = select_channel(data, channel = args.channel) if data.ndim == 3 else data
    print('Number of events to infer:', pulses.shape[0])
    times = predict(args.checkpoint, pulses)

    plt.figure()
    plt.hist(times, bins = args.nbins)
    plt.xlabel('Predicted time (ns)')
    plt.ylabel('Counts')
    plt.show()

    np.savetxt(args.output, times.flatten(), fmt = '%.6f')

else:
    if data.ndim != 3:
        raise ValueError('Coincidence mode (--checkpoint2) needs data of shape (N, M, 2).')
    print('Number of events to infer:', data.shape[0])

    # Coincidence time resolution
    TOF = predict(args.checkpoint, data[:, :, 0]) - predict(args.checkpoint2, data[:, :, 1])

    centroid = calculate_gaussian_center(TOF[np.newaxis, :], nbins = args.nbins, limit = 6)[0]
    (H, x0, FWHM), errors = get_gaussian_params(TOF, centroid, nbins = args.nbins, range = 0.6)

    print(f'Bias: {x0*1000:.2f} ps')
    print(f'CTR resolution (FWHM): {FWHM*1000:.2f} ps')

    plt.figure()
    plot_gaussian(TOF, centroid, nbins = args.nbins, range = 0.6, label = 'Coincidence TOF')
    plt.xlabel(r'$\Delta t$ (ns)')
    plt.ylabel('Counts')
    plt.show()

    np.savetxt(args.output, TOF, fmt = '%.6f')
