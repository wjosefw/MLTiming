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

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def custom_loss_Limit(outputs_0, outputs_1, labels, limit = 1):
    loss = (torch.mean(abs(outputs_0 - outputs_1 - labels)) +
            torch.mean(torch.maximum(torch.tensor(0.0, device = outputs_0.device), 
                                     torch.abs(outputs_1 - outputs_0) - limit)))
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

def custom_loss_with_huber(outputs_0, outputs_1, labels):
    delta = 0.015
    penalty_weight = 1  
    threshold = 1.0  # Threshold for penalty term

    # Compute the predicted difference
    pred_diff = outputs_0 - outputs_1
    error = pred_diff - labels

    # Compute Huber loss
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    regression_loss = torch.where(is_small_error, squared_loss, linear_loss).mean()

    # Penalty term for deviations beyond the threshold
    deviation = torch.abs(pred_diff) - threshold
    penalty = torch.mean(torch.relu(deviation) ** 2)

    # Total loss
    loss = regression_loss + penalty_weight * penalty
    return loss

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def loss_MAE_KAN(outputs_0, outputs_1, labels):
    loss = (torch.mean(abs(outputs_0 - outputs_1 - labels)) +
            torch.sum(torch.relu(-outputs_0)) +
            torch.sum(torch.relu(-outputs_1)))
    return loss

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def loss_MSE_KAN(outputs_0, outputs_1, labels):
    loss = (torch.mean(abs(outputs_0 - outputs_1 - labels)**2) +
            torch.sum(torch.relu(-outputs_0)) +
            torch.sum(torch.relu(-outputs_1)))
    return loss