
import torch.nn as nn
import torch.nn.functional as F


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class ConvolutionalModel(nn.Module):
    def __init__(self, N_points, num_outputs=1):
        super(ConvolutionalModel, self).__init__()

        # Define convolutional blocks 
        self.conv1 = self.conv_block(1, 16, 5) 
        self.conv2 = self.conv_block(16, 32, 5) 
        self.conv3 = self.conv_block(32, 64, 3)
        

        # Calculate flattened size for fully connected layer
        self.flatten_size = 64 * (N_points // 8)  

        # Fully connected layer with output
        self.fc1 = nn.Linear(self.flatten_size, 16)
        self.fc2 = nn.Linear(16, num_outputs)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p = 0.05)

    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, padding = 'same', stride = 1),
            nn.MaxPool1d(kernel_size = 2),
        )

    def forward(self, x):
        # Pass through convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
     
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softplus(x)

        return x
    
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class MLP_Torch(nn.Module):
    def __init__(self, NM = 12, NN = 128, STD_INIT = 0.5):
        super(MLP_Torch, self).__init__()
        self.dense1 = nn.Linear(NM, NN)
        self.dense2 = nn.Linear(NN, NN)
        self.dense3 = nn.Linear(NN, NN)
        self.output = nn.Linear(NN, 1)
        
        # Initialize weights
        nn.init.normal_(self.dense1.weight, mean = 0.0, std = STD_INIT)
        nn.init.normal_(self.dense2.weight, mean = 0.0, std = STD_INIT)
        nn.init.normal_(self.dense3.weight, mean = 0.0, std = STD_INIT)
        nn.init.normal_(self.output.weight, mean = 0.0, std = STD_INIT)
  
    def forward(self, input):
        def forward_single(x):
            x = F.leaky_relu(self.dense1(x))
            x = F.leaky_relu(self.dense2(x))
            x = F.leaky_relu(self.dense3(x))
            x = F.softplus(self.output(x))
            
            return x
        
        out = forward_single(input)
        return out
    

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def build_model(metadata):
    """The model a checkpoint's metadata describes (the dict MLTiming.py saves next to it): model_type
    'CNN' (on the cropped waveform), 'MLPWAVE' (an MLP on the cropped waveform), 'MLP' (an MLP on the
    moments) or 'KAN' (a KAN on the moments; needs efficient-kan, see the README)."""
    model_type = metadata['model_type']
    window = int(metadata['before_samples'] + metadata['after_samples'])
    if model_type == 'CNN':
        return ConvolutionalModel(window)
    if model_type == 'MLPWAVE':
        return MLP_Torch(NM = window, NN = metadata['num_neurons'], STD_INIT = 0.5)
    if model_type == 'MLP':
        return MLP_Torch(NM = metadata['moments_order'], NN = metadata['num_neurons'], STD_INIT = 0.5)
    if model_type == 'KAN':
        try:
            from efficient_kan.src.efficient_kan import KAN
        except ImportError as e:
            raise ImportError("KAN models need efficient-kan in the repository root: "
                              "git clone https://github.com/Blealtan/efficient-kan efficient_kan") from e
        return KAN(metadata['architecture'])
    raise ValueError(f"Unsupported model_type: {model_type}. Use 'CNN', 'MLPWAVE', 'MLP' or 'KAN'.")
