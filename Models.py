
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

def train_loop_KAN(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
    
    loss_list = []
    val_loss_list = []
    test = []
    val = []  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 1e-6)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Move model and test_tensor to the device
    model = model.to(device)
    test_tensor = test_tensor.to(device)

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad()
            
            outputs_0 = model(inputs[:, :, 0])
            outputs_1 = model(inputs[:, :, 1])

            loss = loss_MAE_KAN(outputs_0, outputs_1, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        loss_list.append(running_loss / len(train_loader))

        print(f'EPOCH {epoch + 1}:')
        print(f'LOSS train {running_loss / len(train_loader)}')

        with torch.no_grad():
            test_epoch = model(test_tensor)
            test.append(np.squeeze(test_epoch.cpu().detach().numpy()))

            val_loss = 0
            val_stack = []  # List to hold val_0 and val_1 pairs
            
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device) 
                val_0 = model(val_data[:, :, 0])
                val_1 = model(val_data[:, :, 1])
                val_loss += loss_MAE_KAN(val_0, val_1, val_labels)

                # Stack val_0 and val_1 along the last dimension
                val_stack.append(np.stack((np.squeeze(val_0.cpu().detach().numpy()), np.squeeze(val_1.cpu().detach().numpy())), axis = -1))

            # Combine all batches into a single array for this epoch
            epoch_val = np.concatenate(val_stack, axis = 0)  
            val.append(epoch_val)  
            
            val_loss_list.append(val_loss.item() / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

    if save:
        torch.save(model.state_dict(), name)

    return (
        np.array(loss_list, dtype = 'object'), 
        np.array(val_loss_list, dtype = 'object'), 
        np.array(test, dtype = 'object'), 
        np.array(val, dtype = 'object')
    )

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class ConvolutionalModel_Threshold(nn.Module):
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
    