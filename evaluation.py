import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def E_correlation(SR, GT):
    """
    SR ,GT is flattened
    """
    return np.corrcoef(SR,GT)[0,1]


def get_mean_absolute_error(SR,GT):
    """
    SR, GT [batch, W, H, D]
    """
    return np.abs(SR-GT).mean()


def get_mean_relative_absolute_deviation(SR, GT):
    """
    SR, GT [batch, W, H, D]
    """
    MRE = (np.abs(SR-GT)+0.001)/(GT+0.001)
    return MRE.mean() 

def get_MSE_PSNR(SR, GT):
    """
    SR, GT [batch, W, H, D]
    """
    MSE = ((SR-GT)**2 ).mean()
    MAX = GT.max()
    PSNR = 10* np.log10(MAX**2/MSE)
    return MSE, PSNR


def tissue_error(seg, SR, GT):
    """
    SR, GT [batch, W, H, D]
    """
    # print(seg.shape, GT.shape, seg.shape)
    maxnum = seg.max()
    tissue_MAE = []
    for i in range(maxnum+1):
        indice = (seg==i)
        tmp = np.abs(SR[indice]-GT[indice]).mean()
        tissue_MAE.append(tmp)
    return tissue_MAE


def tissue_MRE(seg, SR, GT):
    """
    SR, GT [batch, W, H, D]
    """
    maxnum = seg.max()
    tissue_MRE = []
    for i in range(maxnum+1):
        indice = (seg==i)
        MRE = (np.abs(SR[indice]-GT[indice])+0.001)/(GT[indice]+0.001)
        tmp = MRE.mean()
        tissue_MRE.append(tmp)
    return tissue_MRE


def value_error( SR, GT):
    valuerange = [0,0.2,0.7,1.2,50]
    tissue_MAE = []
    for i in range(4):
        indice = (GT>=valuerange[i]) & (GT<valuerange[i+1])
        tmp = np.abs(SR[indice]-GT[indice]).mean()
        tissue_MAE.append(tmp)
    return tissue_MAE


def get_accuracy(SR,GT):
    corr = (SR==GT).sum()
    tensor_size = SR.shape[0]*SR.shape[1]*SR.shape[2]*SR.shape[3]
    acc = float(corr)/float(tensor_size)
    return acc


def get_DC(SR, GT, area):
    # DC : Dice Coefficient
    indice1 = (SR == area).astype('int')
    indice2 = (GT == area).astype('int')
    Inter = ((indice1+indice2)==2).sum()
    DC = float(2*Inter)/(indice1.sum()+indice2.sum() + 1e-6)
    return DC


def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS
