
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from Losses import custom_loss_MAE, custom_loss_bounded, custom_loss_MSE, custom_loss_Threshold
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

    
class ConvolutionalModel(nn.Module):
    def __init__(self, N_points):
        super(ConvolutionalModel, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((1, 2))  # Reduces width by half
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1, stride = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((1, 2))  # Reduces width by half
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, stride = 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d((1, 2))  # Reduces width by half

        # Calculate the flattened size after convolutions and pooling
        self.flatten_size = 32 * (N_points // 8)  # Adjust according to pooling layers
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(self.flatten_size, 1)
        
        # Optional Dropout for regularization
        self.dropout = nn.Dropout(p = 0.05)  # Dropout with 5% probability

    def forward(self, x):
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.pool1(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.pool2(x)

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.pool3(x)
        
        # Flattening the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully Connected Layer with Dropout
        x = self.dropout(x)
        x = self.fc1(x)
     
        return x

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_convolutional(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
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


    if save:
        torch.save(model.state_dict(), name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype = 'object')
    test = np.array(test, dtype = 'object')
    val = np.array(val_loss_list, dtype = 'object')
    
    return loss_array, test, val


  
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


def train_loop_MLP(model, optimizer,  train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
    """
    Args:
    model (torch.nn.Module): The model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer used to adjust the model's parameters.
    train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
    val_loader (torch.utils.data.DataLoader): DataLoader containing the validation data.
    test_tensor (torch.Tensor): Tensor used for testing.
    EPOCHS (int, optional): The number of epochs to train the model. Default is 75.
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
       torch.save(model.state_dict(), name)

    

    return np.array(loss_list, dtype = 'object'), np.array(val_loss_list, dtype = 'object'), np.array(test, dtype = 'object')

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------




def train_loop_KAN(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, checkpoint = 15, name = 'model', save = False):
    
    loss_list = []
    val_loss_list = []
    test = []
    val = []  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)
    
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

            loss = custom_loss_MAE(outputs_0, outputs_1, labels)
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
                val_loss += custom_loss_MAE(val_0, val_1, val_labels)

                # Stack val_0 and val_1 along the last dimension
                val_stack.append(np.stack((np.squeeze(val_0.cpu().detach().numpy()), np.squeeze(val_1.cpu().detach().numpy())), axis = -1))

            # Combine all batches into a single array for this epoch
            epoch_val = np.concatenate(val_stack, axis = 0)  
            val.append(epoch_val)  
            
            val_loss_list.append(val_loss.item() / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

        if save and epoch % checkpoint == 0:
            torch.save(model.state_dict(), f'{name}_{epoch}')

    return (
        np.array(loss_list, dtype = 'object'), 
        np.array(val_loss_list, dtype = 'object'), 
        np.array(test, dtype = 'object'), 
        np.array(val, dtype = 'object')
    )

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def train_loop_KAN_bounded(model, optimizer, train_loader, val_loader, low_limit, high_limit, test_tensor, EPOCHS=75, checkpoint = 50, name = 'model', save = False):
    loss_list = []
    val_loss_list = []
    test = []
    val = []  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            
            outputs_0 = model(inputs[:, :, 0])
            outputs_1 = model(inputs[:, :, 1])

            loss = custom_loss_bounded(outputs_0, outputs_1, labels, low_limit, high_limit)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        loss_list.append(running_loss / len(train_loader))

        print(f'EPOCH {epoch + 1}:')
        print(f'LOSS train {running_loss / len(train_loader)}')

        with torch.no_grad():
            test_epoch = model(test_tensor)
            test.append(np.squeeze(test_epoch.detach().numpy()))

            val_loss = 0
            val_stack = []  # List to hold val_0 and val_1 pairs
            
            for val_data, val_labels in val_loader:
                val_0 = model(val_data[:, :, 0])
                val_1 = model(val_data[:, :, 1])
                val_loss += custom_loss_bounded(val_0, val_1, val_labels, low_limit, high_limit)

                # Stack val_0 and val_1 along the last dimension
                val_stack.append(np.stack((np.squeeze(val_0.detach().numpy()), np.squeeze(val_1.detach().numpy())), axis = -1))

            # Combine all batches into a single array for this epoch
            epoch_val = np.concatenate(val_stack, axis = 0)  
            val.append(epoch_val)  
            
            val_loss_list.append(val_loss / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

        if save and epoch % checkpoint == 0:
            torch.save(model.state_dict(), f'{name}_{epoch}')

    return (
        np.array(loss_list, dtype = 'object'), 
        np.array(val_loss_list, dtype = 'object'), 
        np.array(test, dtype = 'object'), 
        np.array(val, dtype = 'object')
    )


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def train_loop_T_Ref(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, checkpoint = 15, name = 'model', save = False):
    loss_list = []
    val_loss_list = []
    test = []
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mse_loss = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        loss_list.append(running_loss / len(train_loader))

        print(f'EPOCH {epoch + 1}:')
        print(f'LOSS train {running_loss / len(train_loader)}')

        with torch.no_grad():
            test_epoch = model(test_tensor)
            test.append(np.squeeze(test_epoch.detach().numpy()))

            val_loss = 0
            for val_data, val_labels in val_loader:
                val_output = model(val_data)
                val_loss += mse_loss(val_output, val_labels)
            val_loss_list.append(val_loss / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

        if save and epoch % checkpoint == 0:
            torch.save(model.state_dict(), f'{name}_{epoch}')

    return np.array(loss_list, dtype = 'object'), np.array(val_loss_list, dtype = 'object'), np.array(test, dtype = 'object')


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())