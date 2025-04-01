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


## 🚀 Features


## 🛠️ Installation

Begin by cloning the entire repository and navigating to the project directory:

```bash
# Clone the repo
git clone https://github.com/wjosefw/MLTiming.git
```

This is a Python-based project. Creating a conda environment is recommended to manage the dependencies. If conda is not already installed, install it from the official site. To make and activate the conda environment with the necessary packages, run:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate DT_env
```

