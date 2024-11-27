
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from Losses import custom_loss_MAE, custom_loss_bounded, custom_loss_MSE, custom_loss_Limit, custom_loss_with_huber, loss_MAE_KAN, loss_MSE_KAN
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_convolutional(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_tensor = test_tensor.to(device)

    loss_list = []
    val_loss_list = []
    test = []
    val_list = []

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 1e-6)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
           
            # Zero gradients
            optimizer.zero_grad()

            # Make predictions for this batch 
            outputs_0 = model(inputs[:, None, :, 0])
            outputs_1 = model(inputs[:, None, :, 1])
            
            # Compute the loss and its gradients
            loss = custom_loss_with_huber(outputs_0, outputs_1, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Calculate average loss per epoch
        avg_loss_epoch = running_loss / len(train_loader)  
        loss_list.append(avg_loss_epoch)

        print(f'EPOCH {epoch + 1}: LOSS train {avg_loss_epoch}')

        # Calculate predictions on test_tensor
        model.eval()
        with torch.no_grad():
            test_epoch = model(test_tensor[:, None, :])
            test.append(np.squeeze(test_epoch.cpu().numpy()))

            val_loss = 0.0
            val_stack = []  # List to hold val_0 and val_1 pairs

            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                
                mask = val_labels != 0 
                val_0 = model(val_data[:, None, :, 0])
                val_1 = model(val_data[:, None, :, 1])
                val_loss += custom_loss_with_huber(val_0[mask], val_1[mask], val_labels[mask]).item()
        
                # Stack val_0 and val_1 along the last dimension
                val_stack.append(np.stack((np.squeeze(val_0.cpu().detach().numpy()), np.squeeze(val_1.cpu().detach().numpy())), axis = -1))

            # Combine all batches into a single array for this epoch
            epoch_val = np.concatenate(val_stack, axis = 0)  
            val_list.append(epoch_val)  
        
            val_loss_list.append(val_loss / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')


    if save:
        torch.save(model.state_dict(), name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype = 'object')
    test = np.array(test, dtype = 'object')
    val_loss = np.array(val_loss_list, dtype = 'object')
    val = np.array(val_list, dtype = 'object')
    
    return loss_array, test, val_loss, val

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
            loss = custom_loss_with_huber(outputs_0, outputs_1, labels)
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
                val_loss += custom_loss_with_huber(val_0, val_1, val_labels)
            val_loss_list.append(val_loss.cpu().numpy() / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')


    # Save the model at the specified checkpoint frequency if 'save' is True
    if save:
       torch.save(model.state_dict(), name)

    

    return np.array(loss_list, dtype = 'object'), np.array(val_loss_list, dtype = 'object'), np.array(test, dtype = 'object')

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

            loss = custom_loss_with_huber(outputs_0, outputs_1, labels)
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
                val_loss += custom_loss_with_huber(val_0, val_1, val_labels)

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

def train_loop_KAN_with_target(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
    
    loss_list = []
    val_loss_list = []
    test = []
    val_list = []  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 1e-6)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Move model and test_tensor to the device
    model = model.to(device)
    test_tensor = test_tensor.to(device)


    # Define MSE loss
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
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
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device) 
                val = model(val_data)
                val_loss += loss_fn(val, val_labels)
                val_list.append(val.cpu().detach().numpy())  
            
            val_loss_list.append(val_loss.item() / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

    if save:
        torch.save(model.state_dict(), name)

    return (
        np.array(loss_list, dtype = 'object'), 
        np.array(val_loss_list, dtype = 'object'), 
        np.array(test, dtype = 'object'), 
        np.array(val_list, dtype = 'object')
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

def train_loop_convolutional_Threshold(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_tensor = test_tensor.to(device)
    
    loss_list = []
    val_loss_list = []
    test = []

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 1e-6)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()

            # Make predictions for this batch 
            outputs_0 = model(inputs[:, None, :, :, 0])
            outputs_1 = model(inputs[:, None, :, :, 1])
            
            # Compute the loss and its gradients
            loss = custom_loss_with_huber(outputs_0, outputs_1, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Calculate average loss per epoch
        avg_loss_epoch = running_loss / len(train_loader)  
        loss_list.append(avg_loss_epoch)

        print(f'EPOCH {epoch + 1}: LOSS train {avg_loss_epoch}')

        # Calculate predictions on test_tensor
        model.eval()
        with torch.no_grad():
            test_epoch = model(test_tensor[:, None, :, :])
            test.append(np.squeeze(test_epoch.cpu().numpy()))

            val_loss = 0.0
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                mask = val_labels != 0 
                val_0 = model(val_data[:, None, :, :, 0])
                val_1 = model(val_data[:, None, :, :, 1])
                val_loss += custom_loss_with_huber(val_0[mask], val_1[mask], val_labels[mask]).item()
        val_loss_list.append(val_loss / len(val_loader))
        print(f'LOSS val {val_loss / len(val_loader)}')


    if save:
        torch.save(model.state_dict(), name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype = 'object')
    test = np.array(test, dtype = 'object')
    val = np.array(val_loss_list, dtype = 'object')
    
    return loss_array, test, val