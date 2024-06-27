from kan import *
import numpy as np

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def train_loop_KAN(model, optimizer, train_loader, test_tensor, EPOCHS = 75):
    """
    Train a KAN Neural Network model for a specified number of epochs.

    Args:
    model: The KAN model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer used to adjust the model's parameters.
    train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
    test_tensor (torch.Tensor): Tensor used for testing/validation.
    EPOCHS (int, optional): The number of epochs to train the model. Default is 75.

    Returns:
    tuple: A tuple containing:
        - loss_array (numpy.ndarray): Array of average losses per epoch during training.
        - test (numpy.ndarray): Array of model predictions on the test_tensor after each epoch.

    The function trains the model by iterating over the training data for a specified number of epochs.
    It calculates the loss, performs backpropagation, and updates the model parameters. After each epoch,
    it evaluates the model on the test_tensor and stores the predictions and loss values.
    """

    epoch_number = 0
    loss_list = []
    test = []

    for epoch in range(EPOCHS):

        running_loss = 0.
        avg_loss_epoch = 0.
    
        for i, data in enumerate(train_loader):
            
            # Every data instance is an input + label pair
            inputs, labels = data
        
            # Zero your gradients for every batch!
            optimizer.zero_grad()
        
            # Make predictions for this batch
            outputs_0 = model(inputs[:,:,0])
            outputs_1 = model(inputs[:,:,1])

            # Compute the loss and its gradients
            loss = torch.mean((outputs_0 - outputs_1 - labels)**2) + torch.sum(torch.relu(-outputs_0)) + torch.sum(torch.relu(-outputs_1))
            loss.backward()
        
            # Adjust learning weights
            optimizer.step()
        
            # Gather data and report
            running_loss += loss.item()

    
        avg_loss_epoch = running_loss / int(i) # loss per batch        
        loss_list.append(avg_loss_epoch)
        
        print('EPOCH {}:'.format(epoch_number + 1))
        print('LOSS train {}'.format(avg_loss_epoch))
        
        # Calculate predictions
        test_epoch = model(test_tensor)
        test.append(np.squeeze(test_epoch.detach().numpy()))

        epoch_number += 1

    # Turn array to list
    loss_array = np.array(loss_list, dtype = 'object')
    test = np.array(test, dtype = 'object')

    return loss_array, test


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def n_params_KAN(architecture, grid_size):
    multiplication_array = np.zeros((len(architecture)-1))
    for i in range(len(architecture) - 1):
        multiplication_array[i] = architecture[i]*architecture[i+1]
    
    n_params = grid_size*np.sum(multiplication_array)
    return int(n_params)


def n_params_KAN(architecture, grid_size):
    """
    Calculate the number of parameters for a given KAN neural network architecture.

    Args:
    architecture (list of int): A list where each element represents the number of neurons in each layer of the neural network.
    grid_size (int): A scalar value that represents the grid size used in the KAN model.

    Returns:
    int: The total number of parameters in the KAN model.

    The function calculates the number of parameters by multiplying the number of neurons between consecutive layers
    and summing these products, then multiplying the sum by the grid size.
    """
    # Initialize an array to hold the products of neurons between consecutive layers
    multiplication_array = np.zeros((len(architecture) - 1))
    
    # Calculate the product of neurons between each consecutive layer
    for i in range(len(architecture) - 1):
        multiplication_array[i] = architecture[i] * architecture[i + 1]
    
    # Sum the products and multiply by the grid size to get the total number of parameters
    n_params = grid_size * np.sum(multiplication_array)
    
    return int(n_params)