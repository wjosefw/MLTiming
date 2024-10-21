import torch



#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def custom_loss_MAE(outputs_0, outputs_1, labels):
    loss = (torch.mean(abs(outputs_0 - outputs_1 - labels)) +
            torch.sum(torch.relu(-outputs_0)) +
            torch.sum(torch.relu(-outputs_1)))
    return loss

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def custom_loss_MSE(outputs_0, outputs_1, labels):
    loss = (torch.mean((outputs_0 - outputs_1 - labels) ** 2) +  
            torch.sum(torch.relu(-outputs_0)) +
            torch.sum(torch.relu(-outputs_1)))
    return loss


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


def custom_loss_Threshold(outputs_0, outputs_1, labels):
    loss = (torch.mean((outputs_0 - outputs_1 - labels) ** 2) +  
            torch.sum(torch.relu(-outputs_0)) +
            torch.sum(torch.relu(-outputs_1)) +
            torch.mean(torch.maximum(torch.tensor(0.0, device=outputs_0.device), 
                                     torch.abs(outputs_1 - outputs_0) - 1)))
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


