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

### OPTION A: Quickstart

In order to get a fast and simple glance on how to use MLTiming go to the quickstart notebook [Link to Notebook](Quickstart_MLTiming.ipynb)

### OPTION B: Inference on your data

If you want to perform directly perform inference on your data, utilize the inference.py [Link to Script](inference.py) script which outputs a .txt file of the time predictions. Data should be saved in a numpy array of shape (N,M) where N is the number of events and M is the number of time points in the signal. You can choose between the available CNN, MLP, MLPWAVE and KAN trained models. For example:

```bash
python inference.py --data path_to_your_data --model CNN
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
2. Gross adjusment
3. Fixed window
3. Threshold

## Trained Models 🧠

* AG_model: Implementation of MLTiming for the CNN architecture.
* KAN_AG_model: Implementation of MLTiming for the KAN architecture.
* MLP_AG_model: Implementation of MLTiming for the MLP architecture with caclculated moments as inputs.
* MLPWAVE_AG_model: Implementation of MLTiming for the MLP architecture with cropped waveform as input.

## Gross Adjusments

It includes scripts to train ML models for timing in radiation detectors using MLTiming.

* config_Gross_Adjustment.py: Configuration file for training hyperamaters.
* MLTiming_Ajuste_grueso_Single_Det.py: Train a single detector for timing using its own waveforms.
* MLTiming_Ajuste_grueso_Conv.py: Train two detectors (separately but in the same script) and evualuate Coincidence time resolution directly for the CNN or MLPWAVE implementation
* MLTiming_Ajuste_grueso.py: Train two detectors (separately but in the same script) and evualuate Coincidence time resolution directly for the MLP or KAN implementation:
* test_model_Ajuste_grueso.py: Test trained models on new data. The script is built for testing Coincidence time resolution so data from two detectors is needed. If you just want to perform time inference on a single detectore use the 'inference.py' script referenced in the usage section.