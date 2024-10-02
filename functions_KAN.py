import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def custom_loss(outputs_0, outputs_1, labels):
    loss = (torch.mean(abs(outputs_0 - outputs_1 - labels)) +
            torch.sum(torch.relu(-outputs_0)) +
            torch.sum(torch.relu(-outputs_1)))
    return loss


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def custom_loss_bounded(outputs_0, outputs_1, labels, lower_limit, upper_limit):
    # Compute the main loss term
    loss = torch.mean(abs(outputs_0 - outputs_1 - labels))

    # Create tensors filled with the limit values of the same shape as outputs
    lower_limit_tensor = torch.full_like(outputs_0, lower_limit)
    upper_limit_tensor = torch.full_like(outputs_0, upper_limit)

    # Calculate penalty for outputs below the lower limit
    penalty_lower_0 = torch.sum(torch.relu(lower_limit_tensor - outputs_0))
    penalty_lower_1 = torch.sum(torch.relu(lower_limit_tensor - outputs_1))
    loss += penalty_lower_0 + penalty_lower_1

    # Calculate penalty for outputs above the upper limit
    penalty_upper_0 = torch.sum(torch.relu(outputs_0 - upper_limit_tensor))
    penalty_upper_1 = torch.sum(torch.relu(outputs_1 - upper_limit_tensor))
    loss += penalty_upper_0 + penalty_upper_1

    return loss
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def train_loop_KAN(model, optimizer, train_loader, val_loader, test_tensor, EPOCHS=75, checkpoint=15, name='model', save=False):
    loss_list = []
    val_loss_list = []
    test = []
    val = []  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
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

            loss = custom_loss(outputs_0, outputs_1, labels)
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
                val_loss += custom_loss(val_0, val_1, val_labels)

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