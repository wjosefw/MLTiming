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

If you want to perform directly perform inference on your data, utilize the inference.py [Link to Notebook](inference.py) script which outputs a .txt file of the time predictions. Data should be saved in a numpy array of shape (N,M) where N is the number of events and M is the number of time points in the signal. You can choose between the available CNN, MLP and KAN trained models.

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


