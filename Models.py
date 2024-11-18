
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

        # Define convolutional blocks with batch normalization and ReLU
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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_index=None):
        self.model.eval()
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)
        if target_index is None:
            target_index = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_index].backward()

        # Compute weights
        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()
        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling on gradients

        # Compute Grad-CAM
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam

