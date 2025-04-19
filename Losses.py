import torch

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def custom_loss_MAE(outputs_0, outputs_1, labels):
    loss = torch.mean(abs(outputs_0 - outputs_1 - labels)) 
    return loss

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def custom_loss_MSE(outputs_0, outputs_1, labels):
    loss = torch.mean((outputs_0 - outputs_1 - labels) ** 2) 
    return loss
