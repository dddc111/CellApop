import torch
import torch.nn.functional as F

def soft_bce_loss(y_pred, y_true, epsilon=0.2):
    """
    Calculate Soft BCE Loss
    Args:
        y_pred: predicted probabilities (N, H, W)
        y_true: ground truth labels (N, H, W)
        epsilon: smoothing parameter, default 0.2
    """
    # Label smoothing
    y_smooth = (1 - epsilon) * y_true + epsilon / 2
    
    # Calculate BCE loss
    bce_loss = -(y_smooth * torch.log(y_pred + 1e-8) + 
                 (1 - y_smooth) * torch.log(1 - y_pred + 1e-8))
    
    return bce_loss.mean()

def dice_loss(y_pred, y_true, smooth=1e-8):
    """
    Calculate Dice Loss
    Args:
        y_pred: predicted probabilities (N, H, W)
        y_true: ground truth labels (N, H, W)
        smooth: smoothing factor to prevent division by zero
    """
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate intersection and union
    intersection = (y_pred_flat * y_true_flat).sum()
    union = y_pred_flat.sum() + y_true_flat.sum()
    
    # Calculate Dice coefficient
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    
    # Dice Loss = 1 - Dice coefficient
    return 1 - dice_coeff

def mask_loss(y_pred, y_true, alpha=0.7, beta=0.3, epsilon=0.2):
    """
    Calculate Combined Mask Loss
    Args:
        y_pred: predicted probabilities (N, H, W)
        y_true: ground truth labels (N, H, W)
        alpha: weight for Soft BCE Loss, default 0.7
        beta: weight for Dice Loss, default 0.3
        epsilon: label smoothing parameter, default 0.2
    """
    # Ensure predictions are in range [0,1]
    y_pred = torch.sigmoid(y_pred)
    
    # Calculate Soft BCE Loss
    loss_soft_bce = soft_bce_loss(y_pred, y_true, epsilon)
    
    # Calculate Dice Loss
    loss_dice = dice_loss(y_pred, y_true)
    
    # Combine losses
    total_loss = alpha * loss_soft_bce + beta * loss_dice
    
    return total_loss, loss_soft_bce, loss_dice


def cross_entropy_loss(y_pred, y_true):
    """
    Calculate Cross-Entropy Loss for multi-class segmentation
    Args:
        y_pred: predicted logits (N, C, H, W) where C is number of classes
        y_true: ground truth labels (N, H, W) with class indices
    """
    # y_pred shape: (N, C, H, W)
    # y_true shape: (N, H, W)
    N, C, H, W = y_pred.shape
    
    # Reshape for cross entropy calculation
    y_pred_flat = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (N*H*W, C)
    y_true_flat = y_true.view(-1).long()  # (N*H*W,)
    
    # Calculate cross entropy
    ce_loss = F.cross_entropy(y_pred_flat, y_true_flat, reduction='mean')
    
    return ce_loss

def dice_loss_multiclass(y_pred, y_true, smooth=1e-8):
    """
    Calculate Dice Loss for multi-class segmentation
    Args:
        y_pred: predicted logits (N, C, H, W) where C is number of classes
        y_true: ground truth labels (N, H, W) with class indices
        smooth: smoothing factor to prevent division by zero
    """
    N, C, H, W = y_pred.shape
    
    # Convert predictions to probabilities
    y_pred_prob = F.softmax(y_pred, dim=1)  # (N, C, H, W)
    
    # Convert ground truth to one-hot encoding
    y_true_onehot = F.one_hot(y_true.long(), num_classes=C).permute(0, 3, 1, 2).float()  # (N, C, H, W)
    
    dice_loss_total = 0.0
    
    # Calculate Dice loss for each class
    for c in range(C):
        y_pred_c = y_pred_prob[:, c, :, :].contiguous().view(-1)  # (N*H*W,)
        y_true_c = y_true_onehot[:, c, :, :].contiguous().view(-1)  # (N*H*W,)
        
        # Calculate intersection and union
        intersection = (y_pred_c * y_true_c).sum()
        union = y_pred_c.sum() + y_true_c.sum()
        
        # Calculate Dice coefficient for this class
        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        
        # Add to total loss
        dice_loss_total += (1 - dice_coeff)
    
    # Average over all classes
    return dice_loss_total / C

def fine_tune_loss(y_pred, y_true, gamma=0.7, epsilon=0.3):
    """
    Calculate Combined Fine-tune Loss (CE + Dice)
    Args:
        y_pred: predicted logits (N, C, H, W) where C is number of classes
        y_true: ground truth labels (N, H, W) with class indices
        gamma: weight for Cross-Entropy Loss, default 0.7
        epsilon: weight for Dice Loss, default 0.3
    """
    # Calculate Cross-Entropy Loss
    ce_loss = cross_entropy_loss(y_pred, y_true)
    
    # Calculate Dice Loss
    dice_loss_val = dice_loss_multiclass(y_pred, y_true)
    
    # Combine losses
    total_loss = gamma * ce_loss + epsilon * dice_loss_val
    
    return total_loss, ce_loss, dice_loss_val