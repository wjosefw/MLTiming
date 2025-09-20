import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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