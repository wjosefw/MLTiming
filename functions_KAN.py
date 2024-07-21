from kan import *
import numpy as np

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def train_loop_KAN(model, optimizer,  train_loader, test_tensor, limit = 5, EPOCHS=75, checkpoint=15, name='model', save=False):
    """
    Train a KAN Neural Network model for a specified number of epochs.

    Args:
    model (torch.nn.Module): The KAN model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer used to adjust the model's parameters.
    train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
    test_tensor (torch.Tensor): Tensor used for testing/validation.
    EPOCHS (int, optional): The number of epochs to train the model. Default is 75.
    checkpoint (int, optional): The frequency (in epochs) at which the model is saved. Default is 15.
    name (str, optional): Base name for the saved model files. Default is 'model'.
    save (bool, optional): Whether to save the model at specified checkpoints. Default is False.

    Returns:
    tuple: A tuple containing:
        - loss_array (numpy.ndarray): Array of average losses per epoch during training.
        - test (numpy.ndarray): Array of model predictions on the test_tensor after each epoch.

    The function trains the model by iterating over the training data for a specified number of epochs.
    It calculates the loss, performs backpropagation, and updates the model parameters. After each epoch,
    it evaluates the model on the test_tensor and stores the predictions and loss values.
    If the 'save' parameter is set to True, the model is saved at the specified checkpoint intervals.
    """

    loss_list = []
    test = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        avg_loss_epoch = 0.0

        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch for both channels
            outputs_0 = model(inputs[:, :, 0])
            outputs_1 = model(inputs[:, :, 1])

            # Compute the loss and its gradients
            loss = torch.mean((outputs_0 - outputs_1 - labels) ** 2) + torch.sum(torch.relu(-outputs_0)) + torch.sum(torch.relu(-outputs_1))

            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

        # Calculate average loss per epoch
        avg_loss_epoch = running_loss / int(i)  # loss per batch
        loss_list.append(avg_loss_epoch)

        print('EPOCH {}:'.format(epoch + 1))
        print('LOSS train {}'.format(avg_loss_epoch))

        # Calculate predictions on test_tensor
        with torch.no_grad():
            test_epoch = model(test_tensor)
            test.append(np.squeeze(test_epoch.detach().numpy()))

        # Save the model at the specified checkpoint frequency if 'save' is True
        if save:
            if epoch % checkpoint == 0:
                model_name = name + '_' + str(epoch)
                torch.save(model.state_dict(), model_name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype='object')
    test = np.array(test, dtype='object')

    return loss_array, test

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def n_params_KAN(architecture, grid_size, order):
    """
    Calculate the number of parameters for a given KAN neural network architecture.

    Args:
    architecture (list of int): A list where each element represents the number of neurons in each layer of the neural network.
    grid_size (int): A scalar value that represents the grid size used in the KAN model.
    order (int): A scalar value representing the order of the polynomial splines.

    Returns:
    int: The total number of parameters in the KAN model.

    The function calculates the number of parameters by multiplying the number of neurons between consecutive layers
    and summing these products, then multiplying the sum by the grid size plus order.
    """
    # Initialize an array to hold the products of neurons between consecutive layers
    multiplication_array = np.zeros((len(architecture) - 1))
    
    # Calculate the product of neurons between each consecutive layer
    for i in range(len(architecture) - 1):
        multiplication_array[i] = architecture[i] * architecture[i + 1]
    
    # Sum the products and multiply by the grid size plus order to get the total number of parameters
    n_params = (grid_size + order) * np.sum(multiplication_array)
    
    return int(n_params)
