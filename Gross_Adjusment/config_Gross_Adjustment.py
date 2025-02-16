import torch
import numpy as np
from pathlib import Path

# General settings
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
batch_size = 32
epochs = 500
learning_rate = 1e-5
Num_Neurons = 16
normalization_method = 'min-max'
moments_order = 3
architecture = [moments_order, 5, 1, 1]    # KAN architecture

# Data settings
time_step = 0.2  # Signal time step in ns
delay_time = 1   # Max delay to training pulses in ns
nbins = 51       # Number of histogram bins
positions = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) 
step_size = 0.066 # 1 cm is a TOF difference of 66.6 ps 
Theoretical_TOF = step_size*positions
before = 8
after = 5

# Paths
BASE_DIR = Path(__file__).resolve().parent #Get parent directory of the file
DATA_DIR = BASE_DIR.parent / "Pulsos15CM20250130_version2"
MODEL_SAVE_DIR = BASE_DIR.parent / "Trained_Models"
REF_PULSE_SAVE_DIR = BASE_DIR.parent / "predictions"

# Save
save = True
