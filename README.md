<p align="center">
  <img src="/figures/scheme_2 (1).png" alt="Project Logo" width="800"/>
</p>

<h1 align="center">MLTiming</h1>

<p align="center">
  Train neural networks to predict time-of-arrival of gamma pulses using singles!
  <br />
</p>

---

## 📝 Description

MLTiming is a framework that enables user to train machine learning (ML) models to assign accurate timestamps to events measured in radiation detectors training using just the signals detected by the individual detectors, without the need of resorting to other detector signals nor simulations. 

Our approach leverages Pairwise learning techniques to train small NN the deliver accurate time-of-arrival predictions. In this repo three different kind of architectures are available for use (CNN, MLP and KANs) but user are encouraged to use our techniques to develop more specific architectures tailored to their specific needs!

Check out our paper, ["MLTiming: Pairwise-Constrained Learning for Improving Timing Resolution in TOF-PET"](https://doi.org/10.1109/TRPMS.2026.3673922), published in IEEE Transactions on Radiation and Plasma Medical Sciences.


## 🚀 Usage

### OPTION A: Train your own model

Use the [MLTiming.py](MLTiming.py) script to train a model on your own data. Hyperparameters and data paths are set in [Config.py](Config.py). Data should be a numpy array of shape (N,M) for a single detector, or (N,M,2) if you have paired coincidence measurements, where N is the number of events and M is the number of time points in the signal. A small ready-to-use example lives in [sample_data](sample_data) so you can try the workflow immediately. For example:

```bash
# Single-detector data
python MLTiming.py

# Paired coincidence data (N,M,2): pick which detector to train on
python MLTiming.py --channel 0
python MLTiming.py --channel 1
```

Each run saves a checkpoint to `Trained_Models/` together with a `<checkpoint>.json` metadata file recording the architecture and preprocessing settings it was trained with, so inference never needs them re-entered by hand.

#### Window and delay for your detector

Each pulse is cropped around the crossing of `threshold` (10 % of the maximum), and the network trains on virtual pairs: two copies of the pulse, each delayed by a random whole number of samples up to `delay_time`, cut from the full waveform. The crop and the delay set how much of the rising edge the network sees, and timing works best when it sees only its first part. In [Config.py](Config.py) they are given in ns (`before_ns`, `after_ns`, `delay_time`), which carry over to another sampling rate. At start-up [MLTiming.py](MLTiming.py) prints how far up the edge the network sees; the best results were at about 15–18 % of the maximum (LaBr3 + PMT pulses, tested at 0.133, 0.2 and 0.4 ns sampling). For pulses of another shape, adjust `after_ns` and `delay_time` until the printed level falls in that range.

### OPTION B: Inference on your data

Use the [Inference.py](Inference.py) script to run a trained checkpoint on your data, outputting a .txt file of the time predictions. Data should be saved in a numpy array (.npy or .npz with key "data") of shape (N,M), or (N,M,2) for paired coincidence measurements. For example:

```bash
# Single detector
python Inference.py --data path_to_your_data --checkpoint Trained_Models/CNN_model

# Coincidence data: pass both detectors' checkpoints to also get the coincidence
# time resolution (CTR) and bias, on top of each detector's predictions
python Inference.py --data path_to_your_data --checkpoint Trained_Models/CNN_model_dec0 --checkpoint2 Trained_Models/CNN_model_dec1
```

## 🛠️ Installation

Begin by cloning the entire repository and navigating to the project directory:

```bash
# Clone the repo
git clone https://github.com/wjosefw/MLTiming.git
```

This is a Python-based project. Creating a conda environment is recommended to manage the dependencies. To make and activate the conda environment with the necessary packages, run:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate DT_env
```

A GPU speeds up training but is not required. KAN models need [efficient-kan](https://github.com/Blealtan/efficient-kan) cloned into the repository root (the other models do not):

```bash
git clone https://github.com/Blealtan/efficient-kan efficient_kan
```
## 📑 Repository Table of Contents 

1. Trained Models
2. Sample data
3. Deprecated

## Trained Models 🧠

Models are saved to `Trained_Models/` as `{model_type}_model` (single detector), or `{model_type}_model_dec{channel}` when trained with `--channel` on paired coincidence data, where `model_type` is one of:

* CNN: Implementation of MLTiming for the CNN architecture.
* KAN: Implementation of MLTiming for the KAN architecture.
* MLP: Implementation of MLTiming for the MLP architecture with calculated moments as inputs.
* MLPWAVE: Implementation of MLTiming for the MLP architecture with cropped waveform as input.

Each checkpoint is saved alongside a `<checkpoint>.json` file recording the architecture and preprocessing settings it was trained with, which [Inference.py](Inference.py) loads automatically.

## Sample data

[sample_data](sample_data) contains a small, ready-to-use coincidence dataset (`sample_data_pos0_train/val/test.npy`, shape (N,M,2)) so you can try the training and inference workflow end-to-end without supplying your own data first.

## Deprecated

[Deprecated](Deprecated) holds earlier approaches kept for reference (mean-pulse alignment, fixed-window and threshold-based methods). They're not maintained, not part of the recommended workflow above, and no longer run: they need functions that `functions.py` no longer has (such as the paper's pair generator `create_and_delay_pulse_pair`, replaced by `create_virtual_pairs`).