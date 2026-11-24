import math
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
time_step = 0.2  # Time step of your data (ns)
nbins = 51       # Number of histogram bins
threshold = 0.1  # Reference threshold crop pulses (fraction of the maximum)

# Crop window and virtual-pair delays, in ns: they carry over to another sampling rate in ns, not in samples.
# The network should see the rising edge up to ~15-18 % of the maximum; MLTiming.py prints the level for your data.
before_ns = 1.6  # Kept before the threshold crossing (at least ~1.6 ns)
after_ns = 1.0   # Kept after it (~1.0-1.2 ns)
delay_time = 1   # Max delay of each virtual-pair member (ns), comparable to after_ns
before = math.ceil(before_ns / time_step - 1e-9)   # in samples: 8 and 5 at 0.2 ns
after = math.ceil(after_ns / time_step - 1e-9)

# Paths
BASE_DIR = Path(__file__).resolve().parent #Get parent directory of the file
DATA_DIR = BASE_DIR / "Pulsos15CM20250130_version2"
MODEL_SAVE_DIR = BASE_DIR / "Trained_Models"
FIGURES_DIR = BASE_DIR / "figures"

# Indicate path to your training/val data (numpy array of (M,N) dimensions)
train_data_path = "sample_data/sample_data_pos0_train.npy"
validation_data_path = "sample_data/sample_data_pos0_val.npy"
