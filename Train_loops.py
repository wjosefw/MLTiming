import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Callable, Union

from Losses import custom_loss_MSE

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def plot_and_save_loss(
    train_loss: torch.tensor,
    val_loss: torch.tensor,
    name: str = 'loss_plot.png',
) -> None:
    """
    Plots and saves the training and validation loss curves.
    """
    # Convert tensors to numpy arrays (ensure they are on the CPU)
    train_loss = train_loss.cpu().detach().numpy()
    val_loss = val_loss.cpu().detach().numpy()

    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, np.log10(train_loss.astype('float32')), label = 'Training Loss', color='blue')
    plt.plot(epochs, np.log10(val_loss.astype('float32')), label = 'Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(name)
    plt.close()

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  

def train_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    EPOCHS: int = 75,
    model_name: str = 'model',
    model_dir: str = './Trained_Models',
    figure_dir:  str = './figures',
    model_type: Optional[str] = None,
    test_tensor: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Training loop for detector convolutional or alternative models with optional 
    test-time inference.

    Behavior: - If test_tensor is provided: returns (train_loss, val_loss, test_preds, val_predictions). 
    - If test_tensor is None: returns (train_loss, val_loss, val_predictions).
    """

    # Start clock for timing training
    start_time = time.time()

    # Select device (GPU if available, otherwise CPU)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Assign default loss function if none provided
    if loss_fn is None:
        loss_fn = custom_loss_MSE  # type: ignore[name-defined]

    # Ensure model_type is provided
    if model_type is None:
        raise ValueError("Error: 'model_type' cannot be None. Please specify a valid model type: CNN, MLP, MLPWAVE or KAN")

    # Move model to target device
    model.to(device)

    # Track whether test predictions should be computed
    has_test = test_tensor is not None
    if has_test:
        test_tensor = test_tensor.to(device)
        test_preds = torch.zeros((EPOCHS, test_tensor.shape[0]), device=device)

    # Allocate tensors for storing losses
    train_loss_list = torch.zeros(EPOCHS, device=device)
    val_loss_list = torch.zeros(EPOCHS, device=device)

    # Pre-compute total validation size for storage allocation
    val_size = 0
    for batch in val_loader:
        v_inputs, _ = batch
        val_size += v_inputs.shape[0]

    # Tensor for storing validation predictions across epochs
    val_predictions = torch.zeros((EPOCHS, val_size, 2), device=device)

    # ---------------- Training loop ---------------- #
    for epoch in range(EPOCHS):
    
        if model_type in ['CNN', 'MLP', 'MLPWAVE']:  
            model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass depends on model type
            if model_type in ['CNN']:
                y0 = model(inputs[:, None, :, 0])  
                y1 = model(inputs[:, None, :, 1])
            if model_type in ['KAN', 'MLP', 'MLPWAVE']:
                y0 = model(inputs[:, :, 0])
                y1 = model(inputs[:, :, 1])

            loss = loss_fn(y0, y1, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / max(1, len(train_loader))
        train_loss_list[epoch] = train_epoch_loss
        
        # ---------------- Validation & Test ---------------- #
        if model_type in ['CNN', 'MLP', 'MLPWAVE']:
            model.eval()

        with torch.no_grad():
            # Compute test predictions if enabled
            if has_test:
                if model_type in ['CNN']:
                    t = model(test_tensor[:, None, :]).squeeze()
                if model_type in ['KAN', 'MLP', 'MLPWAVE']:
                    t = model(test_tensor).squeeze()
                test_preds[epoch, :] = t

            # Validation forward pass
            val_running = 0.0
            start_idx = 0
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                if model_type in ['CNN']:
                    v0 = model(v_inputs[:, None, :, 0])
                    v1 = model(v_inputs[:, None, :, 1])
                if model_type in ['KAN', 'MLP', 'MLPWAVE']:
                    v0 = model(v_inputs[:, :, 0])
                    v1 = model(v_inputs[:, :, 1])

                val_running += loss_fn(v0, v1, v_labels).item()

                bsz = v0.shape[0]
                val_predictions[epoch, start_idx:start_idx + bsz, 0] = v0.squeeze()
                val_predictions[epoch, start_idx:start_idx + bsz, 1] = v1.squeeze()
                start_idx += bsz

            val_epoch_loss = val_running / max(1, len(val_loader))
            val_loss_list[epoch] = val_epoch_loss
            
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            if not os.path.isdir(figure_dir):
                os.makedirs(figure_dir)
            plot_and_save_loss(train_loss_list[:epoch], val_loss_list[:epoch], name = os.path.join(figure_dir, f'{model_name}_loss_plot.png'))
            print(f'EPOCH {epoch + 1}: LOSS train {train_epoch_loss:.6f}')
            print(f'LOSS val {val_epoch_loss:.6f}')
    
    # Stop clock
    stop_time = time.time()
    print(f'Training time {stop_time - start_time} seconds')
    
    # Save model weights to specified directory
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}'))

    # Return outputs depending on whether test predictions were requested
    if has_test:
        return test_preds.cpu().numpy(), val_predictions.cpu().numpy()
    else:
        return val_predictions.cpu().numpy()
