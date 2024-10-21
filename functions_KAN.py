import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


from Losses import custom_loss_MAE, custom_loss_bounded
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