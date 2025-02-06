
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
    
    # Pre-allocate loss tracking tensors
    loss_list = torch.zeros(EPOCHS, device = device)
    val_loss_list = torch.zeros(EPOCHS, device = device)
    test = torch.zeros((EPOCHS, test_tensor.shape[0]), device = device)
    
    # Determine validation dataset size (assuming fixed size batches)
    val_size = sum(batch[0].shape[0] for batch in val_loader)  # Total number of validation samples
    val_predictions = torch.zeros((EPOCHS, val_size, 2), device = device)  # Preallocate tensor for all predictions

    # Define loss function 
    loss_fn = custom_loss_MSE  

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
            loss = loss_fn(outputs_0, outputs_1, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

        # Store training loss
        loss_list[epoch] = running_loss / len(train_loader)
        print(f'EPOCH {epoch + 1}: LOSS train {loss_list[epoch].item()}')

        # Calculate predictions on test_tensor
        model.eval()
        with torch.no_grad():
            test_epoch = model(test_tensor[:, None, :])
            test[epoch, :] = test_epoch.squeeze(-1)
            val_loss = 0.0
            start_idx = 0  # Track position in val_predictions tensor

            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                val_0 = model(val_data[:, None, :, 0])
                val_1 = model(val_data[:, None, :, 1])
                val_loss += loss_fn(val_0, val_1, val_labels).item()

                batch_size = val_0.shape[0]
                val_predictions[epoch, start_idx:start_idx + batch_size, 0] = val_0.squeeze()
                val_predictions[epoch, start_idx:start_idx + batch_size, 1] = val_1.squeeze()
                start_idx += batch_size  # Update index

        # Store validation loss
        val_loss_list[epoch] = val_loss / len(val_loader)
        print(f'LOSS val {val_loss_list[epoch].item()}')

    if save:
        torch.save(model.state_dict(), name)

    
    return loss_list.cpu().numpy(), val_loss_list.cpu().numpy(), test.cpu().numpy(), val_predictions.cpu().numpy()

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_convolutional_single_det(model, optimizer, train_loader, val_loader, EPOCHS=75, name='model', save=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Pre-allocate loss tracking tensors
    loss_list = torch.zeros(EPOCHS, device=device)
    val_loss_list = torch.zeros(EPOCHS, device=device)

    # Determine validation dataset size (assuming fixed size batches)
    val_size = sum(batch[0].shape[0] for batch in val_loader)  # Total number of validation samples
    val_predictions = torch.zeros((EPOCHS, val_size, 2), device=device)  # Preallocate tensor for all predictions

    # Define loss function
    loss_fn = custom_loss_MSE  

    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Make predictions for this batch
            outputs_0 = model(inputs[:, None, :, 0])
            outputs_1 = model(inputs[:, None, :, 1])

            # Compute loss and gradients
            loss = loss_fn(outputs_0, outputs_1, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Store training loss
        loss_list[epoch] = running_loss / len(train_loader)
        print(f'EPOCH {epoch + 1}: LOSS train {loss_list[epoch].item()}')

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            start_idx = 0  # Track position in val_predictions tensor

            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                val_0 = model(val_data[:, None, :, 0])
                val_1 = model(val_data[:, None, :, 1])
                val_loss += loss_fn(val_0, val_1, val_labels).item()

                batch_size = val_0.shape[0]
                val_predictions[epoch, start_idx:start_idx + batch_size, 0] = val_0.squeeze()
                val_predictions[epoch, start_idx:start_idx + batch_size, 1] = val_1.squeeze()
                start_idx += batch_size  # Update index

        # Store validation loss
        val_loss_list[epoch] = val_loss / len(val_loader)
        print(f'LOSS val {val_loss_list[epoch].item()}')

    if save:
        torch.save(model.state_dict(), name)

    # Convert loss tensors to NumPy arrays
    return loss_list.cpu().numpy(), val_loss_list.cpu().numpy(), val_predictions.cpu().numpy()

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_convolutional_with_target(model, optimizer, train_loader, val_loader, EPOCHS = 75, name = 'model', save = False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    loss_list = []
    val_loss_list = []
  
    # Define loss function
    loss_fn = torch.nn.L1Loss()

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
            outputs = model(inputs[:, None, :])
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
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
            val_loss = 0.0

            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                
                val = model(val_data[:, None, :])
                val_loss += loss_fn(val, val_labels).item()
    
        
            val_loss_list.append(val_loss / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')


    if save:
        torch.save(model.state_dict(), name)

    # Convert lists to numpy arrays
    loss_array = np.array(loss_list, dtype = 'object')
    val_loss = np.array(val_loss_list, dtype = 'object')
    
    return loss_array, val_loss

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_MLP(model, optimizer,  train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_tensor = test_tensor.to(device)
    
    # Pre-allocate loss tracking tensors
    loss_list = torch.zeros(EPOCHS, device = device)
    val_loss_list = torch.zeros(EPOCHS, device = device)
    test = torch.zeros((EPOCHS, test_tensor.shape[0]), device = device)
    
    # Determine validation dataset size (assuming fixed size batches)
    val_size = sum(batch[0].shape[0] for batch in val_loader)  # Total number of validation samples
    val_predictions = torch.zeros((EPOCHS, val_size, 2), device = device)  # Preallocate tensor for all predictions

    # Define loss function
    loss_fn = custom_loss_MSE

    for epoch in range(EPOCHS):
        running_loss = 0.0
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
            loss = loss_fn(outputs_0, outputs_1, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            
            # Accumulate running loss
            running_loss += loss.item()

        # Store training loss
        loss_list[epoch] = running_loss / len(train_loader)
        print(f'EPOCH {epoch + 1}: LOSS train {loss_list[epoch].item()}')

        # Validation step
        model.eval()
        with torch.no_grad():
            test_epoch = model(test_tensor)
            test[epoch, :] = test_epoch.squeeze(-1)

            val_loss = 0.0
            start_idx = 0  # Track position in val_predictions tensor

            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                val_0 = model(val_data[:, :, 0])
                val_1 = model(val_data[:, :, 1])
                val_loss += loss_fn(val_0, val_1, val_labels).item()

                batch_size = val_0.shape[0]
                val_predictions[epoch, start_idx:start_idx + batch_size, 0] = val_0.squeeze()
                val_predictions[epoch, start_idx:start_idx + batch_size, 1] = val_1.squeeze()
                start_idx += batch_size  # Update index

            # Store validation loss
            val_loss_list[epoch] = val_loss / len(val_loader)
            print(f'LOSS val {val_loss_list[epoch].item()}')


    # Save the model at the specified checkpoint frequency if 'save' is True
    if save:
       torch.save(model.state_dict(), name)

    
    return loss_list.cpu().numpy(), val_loss_list.cpu().numpy(), test.cpu().numpy(), val_predictions.cpu().numpy()

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_KAN(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS = 75, name = 'model', save = False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_tensor = test_tensor.to(device)

    # Pre-allocate loss tracking tensors
    loss_list = torch.zeros(EPOCHS, device = device)
    val_loss_list = torch.zeros(EPOCHS, device = device)
    test = torch.zeros((EPOCHS, test_tensor.shape[0]), device = device)
    
    # Determine validation dataset size (assuming fixed size batches)
    val_size = sum(batch[0].shape[0] for batch in val_loader)  # Total number of validation samples
    val_predictions = torch.zeros((EPOCHS, val_size, 2), device = device)  # Preallocate tensor for all predictions
    
    # Define loss function
    loss_fn = custom_loss_MSE 

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad()
            
            outputs_0 = model(inputs[:, :, 0])
            outputs_1 = model(inputs[:, :, 1])

            loss = loss_fn(outputs_0, outputs_1, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
       
        # Store training loss
        loss_list[epoch] = running_loss / len(train_loader)
        print(f'EPOCH {epoch + 1}: LOSS train {loss_list[epoch].item()}')

        # Validation step
        with torch.no_grad():
            test_epoch = model(test_tensor)
            test[epoch, :] = test_epoch.squeeze(-1)

            val_loss = 0.0
            start_idx = 0  # Track position in val_predictions tensor

            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                val_0 = model(val_data[:, :, 0])
                val_1 = model(val_data[:, :, 1])
                val_loss += loss_fn(val_0, val_1, val_labels).item()

                batch_size = val_0.shape[0]
                val_predictions[epoch, start_idx:start_idx + batch_size, 0] = val_0.squeeze()
                val_predictions[epoch, start_idx:start_idx + batch_size, 1] = val_1.squeeze()
                start_idx += batch_size  # Update index

            # Store validation loss
            val_loss_list[epoch] = val_loss / len(val_loader)
            print(f'LOSS val {val_loss_list[epoch].item()}')

    if save:
        torch.save(model.state_dict(), name)

    return loss_list.cpu().numpy(), val_loss_list.cpu().numpy(), test.cpu().numpy(), val_predictions.cpu().numpy()

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_KAN_with_target(model, optimizer, train_loader, val_loader, EPOCHS = 75, name = 'model', save = False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
     # Define loss
    loss_fn = torch.nn.L1Loss()

    # Define scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 1e-6)
    
    loss_list = []
    val_loss_list = []
    

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
        
            val_loss = 0
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device) 
                val = model(val_data)
                val_loss += loss_fn(val, val_labels)
                
            val_loss_list.append(val_loss.item() / len(val_loader))
            print(f'LOSS val {val_loss / len(val_loader)}')

    if save:
        torch.save(model.state_dict(), name)

    return (
        np.array(loss_list, dtype = 'object'), 
        np.array(val_loss_list, dtype = 'object')
    )

