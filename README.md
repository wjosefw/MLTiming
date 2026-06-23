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

[Deprecated](Deprecated) holds earlier approaches kept for reference (mean-pulse alignment, fixed-window and threshold-based methods). They're not maintained and not part of the recommended workflow above.