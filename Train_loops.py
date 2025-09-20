
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Callable, Union



from Losses import custom_loss_MSE

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_convolutional(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    EPOCHS: int = 75,
    name: str = 'model',
    save: bool = False,
    test_tensor: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Training loop for detector convolutional models with optional test-time inference.

    Behavior:
      - If `test_tensor` is provided: returns (train_loss, val_loss, test_preds, val_predictions).
      - If `test_tensor` is None:    returns (train_loss, val_loss, val_predictions).

    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss_fn is None:
        loss_fn = custom_loss_MSE  # type: ignore[name-defined]

    model.to(device)

    has_test = test_tensor is not None
    if has_test:
        test_tensor = test_tensor.to(device)
        test_preds = torch.zeros((EPOCHS, test_tensor.shape[0]), device=device)

    train_loss_list = torch.zeros(EPOCHS, device=device)
    val_loss_list = torch.zeros(EPOCHS, device=device)

    val_size = 0
    for batch in val_loader:
        v_inputs, _ = batch
        val_size += v_inputs.shape[0]

    val_predictions = torch.zeros((EPOCHS, val_size, 2), device=device)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            y0 = model(inputs[:, None, :, 0])
            y1 = model(inputs[:, None, :, 1])

            loss = loss_fn(y0, y1, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / max(1, len(train_loader))
        train_loss_list[epoch] = train_epoch_loss
        print(f'EPOCH {epoch + 1}: LOSS train {train_epoch_loss:.6f}')

        model.eval()
        with torch.no_grad():
            if has_test:
                t = model(test_tensor[:, None, :]).squeeze(-1)
                test_preds[epoch, :] = t

            val_running = 0.0
            start_idx = 0
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                v0 = model(v_inputs[:, None, :, 0])
                v1 = model(v_inputs[:, None, :, 1])

                val_running += loss_fn(v0, v1, v_labels).item()

                bsz = v0.shape[0]
                val_predictions[epoch, start_idx:start_idx + bsz, 0] = v0.squeeze(-1)
                val_predictions[epoch, start_idx:start_idx + bsz, 1] = v1.squeeze(-1)
                start_idx += bsz

            val_epoch_loss = val_running / max(1, len(val_loader))
            val_loss_list[epoch] = val_epoch_loss
            print(f'LOSS val {val_epoch_loss:.6f}')

    if save:
        torch.save(model.state_dict(), name)

    train_np = train_loss_list.cpu().numpy()
    val_np = val_loss_list.cpu().numpy()
    val_pred_np = val_predictions.cpu().numpy()

    if has_test:
        test_np = test_preds.cpu().numpy()
        return train_np, val_np, test_np, val_pred_np
    else:
        return train_np, val_np, val_pred_np

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop_MLP(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    EPOCHS: int = 75,
    name: str = 'model',
    save: bool = False,
    test_tensor: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Training loop for detector MLP models with optional test-time inference.

    Behavior:
      - If `test_tensor` is provided: returns (train_loss, val_loss, test_preds, val_predictions).
      - If `test_tensor` is None:    returns (train_loss, val_loss, val_predictions).

    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss_fn is None:
        loss_fn = custom_loss_MSE  # type: ignore[name-defined]

    model.to(device)

    has_test = test_tensor is not None
    if has_test:
        test_tensor = test_tensor.to(device)
        test_preds = torch.zeros((EPOCHS, test_tensor.shape[0]), device=device)

    train_loss_list = torch.zeros(EPOCHS, device=device)
    val_loss_list = torch.zeros(EPOCHS, device=device)

    val_size = 0
    for batch in val_loader:
        v_inputs, _ = batch
        val_size += v_inputs.shape[0]

    val_predictions = torch.zeros((EPOCHS, val_size, 2), device=device)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            y0 = model(inputs[:, :, 0])
            y1 = model(inputs[:, :, 1])

            loss = loss_fn(y0, y1, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / max(1, len(train_loader))
        train_loss_list[epoch] = train_epoch_loss
        print(f'EPOCH {epoch + 1}: LOSS train {train_epoch_loss:.6f}')

        model.eval()
        with torch.no_grad():
            if has_test:
                t = model(test_tensor).squeeze(-1)
                test_preds[epoch, :] = t

            val_running = 0.0
            start_idx = 0
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                v0 = model(v_inputs[:, :, 0])
                v1 = model(v_inputs[:, :, 1])

                val_running += loss_fn(v0, v1, v_labels).item()

                bsz = v0.shape[0]
                val_predictions[epoch, start_idx:start_idx + bsz, 0] = v0.squeeze(-1)
                val_predictions[epoch, start_idx:start_idx + bsz, 1] = v1.squeeze(-1)
                start_idx += bsz

            val_epoch_loss = val_running / max(1, len(val_loader))
            val_loss_list[epoch] = val_epoch_loss
            print(f'LOSS val {val_epoch_loss:.6f}')

    if save:
        torch.save(model.state_dict(), name)

    train_np = train_loss_list.cpu().numpy()
    val_np = val_loss_list.cpu().numpy()
    val_pred_np = val_predictions.cpu().numpy()

    if has_test:
        test_np = test_preds.cpu().numpy()
        return train_np, val_np, test_np, val_pred_np
    else:
        return train_np, val_np, val_pred_np

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