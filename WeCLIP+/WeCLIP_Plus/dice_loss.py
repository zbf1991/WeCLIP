import torch
import torch.nn.functional as F

def dice_loss(output, target, eps=1e-7):
    eps = 1e-7
    # valid_target = target[target != 255]
    # valid_output = output[target.unsqueeze(1).expand_as(output) != 255]
    target[target==255] = output.shape[1]
    # convert target to onehot
    targ_onehot = torch.eye(output.shape[1]+1).cuda()[target].permute(0,3,1,2).float()
    targ_onehot = targ_onehot[:,:-1,:,:]
    # convert logits to probs
    pred = F.softmax(output, dim=1)

    target_ex = target.unsqueeze(1).expand_as(pred)
    valid_region = torch.ones_like(pred)
    valid_region[target_ex == 255] = 0
    # sum over HW
    inter = (pred*valid_region * targ_onehot).sum(axis=[0,2,3])
    union = (pred*valid_region + targ_onehot*valid_region).sum(axis=[0,2,3])
    # mean over C
    dice = (2. * inter / (union + eps)).mean()
    return 1. - dice
    

class DiceLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, output, targ):
        """
        output is NCHW, targ is NHW
        """
        return dice_loss(output, targ)

    def activation(self, output):
        return F.softmax(output, dim=1)
    
    def decodes(self, output):
        return output.argmax(1)