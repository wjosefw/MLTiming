
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from Losses import custom_loss_MAE
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class ConvolutionalModel(nn.Module):
    def __init__(self, N_points):
        super(ConvolutionalModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((1, 2)) # Output: (batch_size, 8, 1, N_points // 2)
        
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1, stride = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((1, 2))  # Output: (batch_size, 8, 1, N_points // 4)
        
        # Calculate the flattened size after the convolutions and pooling
        self.flatten_size = 16 * (N_points // 4)  # Adjust according to pooling layers
        
        self.fc1 = nn.Linear(self.flatten_size, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
     
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


def train_loop_MLP(model, optimizer,  train_loader, val_loader, test_tensor, EPOCHS = 75, checkpoint = 15, name = 'model', save = False):
    """
    Args:
    model (torch.nn.Module): The model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer used to adjust the model's parameters.
    train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
    val_loader (torch.utils.data.DataLoader): DataLoader containing the validation data.
    test_tensor (torch.Tensor): Tensor used for testing.
    EPOCHS (int, optional): The number of epochs to train the model. Default is 75.
    checkpoint (int, optional): The frequency (in epochs) at which the model is saved. Default is 15.
    name (str, optional): Base name for the saved model files. Default is 'model'.
    save (bool, optional): Whether to save the model at specified checkpoints. Default is False.

    Returns:
    tuple: A tuple containing:
        - loss_array (numpy.ndarray): Array of average losses per epoch during training.
        - test (numpy.ndarray): Array of model predictions on the test_tensor after each epoch.

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_tensor = test_tensor.to(device)
    
    
    loss_list = []
    val_loss_list = []
    test = []

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        avg_loss_epoch = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero your gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch for both channels
            outputs_0 = model(inputs[:, :, 0])
            outputs_1 = model(inputs[:, :, 1])

            # Compute the loss and its gradients
            loss = custom_loss_MAE(outputs_0, outputs_1, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            
            # Accumulate running loss
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Calculate average loss per epoch
        avg_loss_epoch = running_loss / int(i)  # loss per batch
        loss_list.append(avg_loss_epoch)

        print('EPOCH {}:'.format(epoch + 1))
        print('LOSS train {}'.format(avg_loss_epoch))

        # Calculate predictions on test_tensor
        model.eval()
        with torch.no_grad():
            test_epoch = model(test_tensor)
            test.append(np.squeeze(test_epoch.cpu().numpy()))

            val_loss = 0
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_0 = model(val_data[:, :, 0])
                val_1 = model(val_data[:, :, 1])
                val_loss += custom_loss_MAE(val_0, val_1, val_labels)
            val_loss_list.append(val_loss.cpu().numpy() / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')


        # Save the model at the specified checkpoint frequency if 'save' is True
        if save:
            if epoch % checkpoint == 0:
                model_name = name + '_' + str(epoch)
                torch.save(model.state_dict(), model_name)

    

    return np.array(loss_list, dtype = 'object'), np.array(val_loss_list, dtype = 'object'), np.array(test, dtype = 'object')

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def train_loop_convolutional(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, checkpoint = 15, name = 'model', save = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_tensor = test_tensor.to(device)
    
    loss_list = []
    val_loss_list = []
    test = []

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Make predictions for this batch 
            outputs_0 = model(inputs[:, None, None, :, 0])
            outputs_1 = model(inputs[:, None, None, :, 1])

            # Compute the loss and its gradients
            loss = custom_loss_MAE(outputs_0, outputs_1, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            
            # Accumulate running loss
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Calculate average loss per epoch
        avg_loss_epoch = running_loss / (i)  
        loss_list.append(avg_loss_epoch)

        print(f'EPOCH {epoch + 1}: LOSS train {avg_loss_epoch}')

        # Calculate predictions on test_tensor
        model.eval()
        with torch.no_grad():
            test_epoch = model(test_tensor[:,None, None, :])
            test.append(np.squeeze(test_epoch.cpu().numpy()))

            val_loss = 0
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_0 = model(val_data[:, None, None, :, 0])
                val_1 = model(val_data[:, None, None, :, 1])
                val_loss += custom_loss_MAE(val_0, val_1, val_labels)
        val_loss_list.append(val_loss.cpu().numpy() / len(val_loader))
        print(f'LOSS val {val_loss / len(val_loader)}')


        if save and (epoch + 1) % checkpoint == 0:
            model_name = f'{name}_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype = 'object')
    test = np.array(test, dtype = 'object')
    val = np.array(val_loss_list, dtype = 'object')
    
    return loss_array, test, val