import torch
from pathlib import Path

# General settings
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
batch_size = 32
epochs = 750
learning_rate = 1e-5
Num_Neurons = 16
normalization_method = 'min-max'
moments_order = 3
architecture = [moments_order, 5, 1, 1]    # KAN architecture
model_type = 'CNN' # Type of model to train CNN, MLP, MLPWAVE and KAN available

# Data settings
time_step = 0.2  # Time step of your data
delay_time = 1   # Max delay to training pulses in ns
nbins = 51       # Number of histogram bins
before = 8
after = 5
threshold = 0.1  # Reference threshold crop pulses

# Paths
BASE_DIR = Path(__file__).resolve().parent #Get parent directory of the file
DATA_DIR = BASE_DIR / "Pulsos15CM20250130_version2"
MODEL_SAVE_DIR = BASE_DIR / "Trained_Models"
FIGURES_DIR = BASE_DIR / "figures"

# Indicate path to your training/val data (numpy array of (M,N) dimensions)
train_data_path = "sample_data/sample_data_pos0_train.npy"
validation_data_path = "sample_data/sample_data_pos0_val.npy"
