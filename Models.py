
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        self.fc1 = nn.Linear(self.flatten_size, 32)
        self.fc2 = nn.Linear(32, num_outputs)
        
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
            x = F.relu(self.dense1(x))
            x = F.relu(self.dense2(x))
            x = F.relu(self.dense3(x))
            x = self.output(x)
            return x
        
        out = forward_single(input)
        return out
    

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class ConvolutionalModel_Threshold(nn.Module):
    def __init__(self, N_points, num_outputs=1):
        super(ConvolutionalModel_Threshold, self).__init__()

        # Define convolutional blocks with batch normalization and ReLU
        self.conv1 = self.conv_block(1, 16, (2,5))
        self.conv2 = self.conv_block(16, 32, (2,5))
        self.conv3 = self.conv_block(32, 64, (2,3))
        

        # Calculate flattened size for fully connected layer
        self.flatten_size = 2 * 64 * (N_points // 8)  

        # Fully connected layer with output
        self.fc1 = nn.Linear(self.flatten_size, 32)
        self.fc2 = nn.Linear(32, num_outputs)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p = 0.05)

    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = 'same', stride = 1),
            nn.MaxPool2d(kernel_size = (1,2)),
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, padding = 'same', stride = 1)
        )
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size = 1, padding = 0, stride = 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)  # Main path
        skip = self.skip(x) if self.skip else x  # Skip connection
        return out + skip  # Residual addition


class ResNet1D(nn.Module):
    def __init__(self, N_points, num_outputs=1):
        super(ResNet1D, self).__init__()

        # Define residual blocks
        self.block1 = ResidualBlock(1, 16, 5)
        self.pool1 = nn.MaxPool1d(kernel_size = 2)

        self.block2 = ResidualBlock(16, 32, 5)
        self.pool2 = nn.MaxPool1d(kernel_size = 2)

        self.block3 = ResidualBlock(32, 64, 3)
        self.pool3 = nn.MaxPool1d(kernel_size = 2)

        # Calculate flattened size for fully connected layer
        self.flatten_size = 64  * (N_points // 8)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 32)
        self.fc2 = nn.Linear(32, num_outputs)

        # Dropout for regularization
        self.dropout = nn.Dropout(p = 0.05)

    def forward(self, x):
        # Pass through residual blocks
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softplus(x)

        return x
