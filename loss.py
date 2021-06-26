import torch
from torch import nn
from torch.nn import functional as f


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Default softmax function. Operates on channels in 4D tensors.

    x : torch.Tensor

    return: torch.Tensor
    """
    return f.softmax(x)


def log_cosh_dc(pred: torch.Tensor, weights: torch.Tensor, target: torch.Tensor, sigmoid: bool) -> torch.Tensor:
    """
    Log-Cosh Dice Loss.

    prev : torch.Tensor
        Predicted map. 4D tensor, may be not normalized by the sigmoid.

    weights : torch.Tensor
        Pixel wise weights for predicted map. Normalized [0,1] 4D tensor.

    target : torch.Tensor
        Target (ground truth) map. Normalized [0,1] 4D tensor.

    sigmoid : bool
        Normalizes predicted mask with sigmoid function if True.

    return: torch.Tensor
        Loss value. Float tensor.
    """
    dc = dice_loss(pred, weights, target, sigmoid=sigmoid)

    return torch.log((torch.exp(dc) + torch.exp(-dc)) / 2.0)


def bce_loss(pred: torch.Tensor, weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Weighted binary cross entropy loss.

    prev : torch.Tensor
        Predicted map. 4D tensor, has to be normalized by the sigmoid.

    weights : torch.Tensor
        Pixel wise weights for predicted map. Normalized [0,1] 4D tensor.

    target : torch.Tensor
        Target (ground truth) map. Normalized [0,1] 4D tensor.

    return: torch.Tensor
        Loss value. Float tensor.
    """
    return f.binary_cross_entropy(pred, target, weight=weights, reduction="mean")


def bce_digits(pred: torch.Tensor, weights: torch.Tensor, target: torch.Tensor,
               pw: (None, float) = None) -> torch.Tensor:
    """
    Weighted binary cross entropy loss with logits and positive weights.

    prev : torch.Tensor
        Predicted map. 4D tensor, not normalized by the sigmoid.

    weights : torch.Tensor
        Pixel wise weights for predicted map. Normalized [0,1] 4D tensor.

    target : torch.Tensor
        Target (ground truth) map. Normalized [0,1] 4D tensor.

    pw : None or float
        Weight of positive values. If None, pw = 1.

    return: torch.Tensor
        Loss value. Float tensor.
    """
    if not (pw is None):
        pw = torch.ones_like(pred)
    return f.binary_cross_entropy_with_logits(pred, target, weights, pos_weight=pw)


def dice_loss(pred: torch.Tensor, weights: torch.Tensor, target: torch.Tensor,
              epsilon: float = 1e-7, sigmoid: bool = False) -> torch.Tensor:
    """
    Weighted dice loss.

    prev : torch.Tensor
        Predicted map. 4D tensor, may be not normalized by the sigmoid.

    weights : torch.Tensor
        Pixel wise weights for predicted map. Normalized [0,1] 4D tensor.

    target : torch.Tensor
        Target (ground truth) map. Normalized [0,1] 4D tensor.

    epsilon : float
        Stability value. Default is 1e-7.

    sigmoid : bool
        Normalizes predicted mask with sigmoid function if True.

    return: torch.Tensor
        Loss value. Float tensor.
    """
    if sigmoid:
        pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (weights * pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + epsilon) / (
            (weights * pred).sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)))
    return loss.mean()


def dice_coeff(pred: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-7, threshold: float = 0.5, soft_flag: bool = False):
    """
    Dice coeficient for multiclass cases.

    pred : torch.Tensor
        Predicted map. 4D tensor, not normilized by Softmax

    target : torch.Tensor
        Target (ground truth) map. Normalized [0,1] 4D tensor.

    epsilon : float
        Stability value. Default is 1e-7.

    softmax : bool
        Normalizes predicted mask with sigmoid or softmax, depending on dimension.

    return: torch.Tensor
        Loss value. Float tensor.
    """
    num_classes = pred.shape[1]
    if num_classes == 1:
        target = torch.tensor(target, dtype=torch.long)
        true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        if soft_flag:
            pos_prob = torch.sigmoid(pred)
        else:
            pos_prob = pred
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        if soft_flag:
            probas = f.softmax(pred, dim=1)
        else:
            probas = pred

    probas = (probas > threshold).float()
    true_1_hot = true_1_hot.type(probas.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dsc = ((2. * intersection + eps) / (cardinality + eps)).mean()
    return dsc


def dice_coeff_binary(pred: torch.Tensor, target: torch.Tensor, threshold=0.5, epsilon=1e-6,
                      sigmoid=False) -> torch.Tensor:
    if sigmoid:
        pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)
    return dice.mean()


class Focal_Loss(nn.Module):
    def __init__(self, gamma=2.0, loss_type='bce_digits', pw=0.5):
        super(Focal_Loss, self).__init__()
        self.gamma = nn.Parameter(data=torch.tensor(gamma), requires_grad=False)
        self.pw = nn.Parameter(data=torch.tensor(pw), requires_grad=False)

        if loss_type == 'bce':
            self.loss = self.calc_bce
        elif loss_type == 'bce_digits':
            self.loss = self.calc_bce_digits
        elif loss_type == 'ce_digits':
            self.loss = self.calc_ce_digits
            self.pw = nn.Parameter(torch.tensor([1.0, pw, pw]), requires_grad=False)

    def calc_bce(self, pred, weights, target):
        bce = f.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = weights * (1 - pt) ** self.gamma * bce
        return focal.sum()

    def calc_bce_digits(self, pred, weights, target):
        bce = f.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        pw = self.pw * torch.ones_like(pred)
        focal = weights * ((1 - pt) ** self.gamma) * f.binary_cross_entropy_with_logits(pred, target,
                                                                                        reduction="none", pos_weight=pw)
        return focal.sum() / len(pred)

    def calc_ce_digits(self, pred, weights, target, att):
        ce = f.cross_entropy(pred, target.squeeze(1), reduction="none")
        pt = torch.exp(-ce)
        focal_loss = att * weights * ((1 - pt) ** self.gamma) * f.cross_entropy(pred, target.squeeze(1),
                                                                                reduction="none")
        return focal_loss.sum() / len(pred)

    def forward(self, pred, weights, target):  # Input is already after sigmoid
        return self.loss(pred, weights, target)


class Tversky(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0):
        super(Tversky, self).__init__()
        self.gamma = nn.Parameter(data=torch.tensor(gamma), requires_grad=False)
        self.alpha = nn.Parameter(data=torch.tensor(alpha), requires_grad=False)

        self.forward = self.tl

        if gamma > 1.0:
            self.forward = self.tfl

    def tl(self, pred, weights, target):
        pred = weights * torch.sigmoid(pred)

        pred = pred.contiguous()
        target = target.contiguous()

        inter = (pred * target).sum(dim=2).sum(dim=2)
        a_inter = self.alpha * ((1 - target) * pred).sum(dim=2).sum(dim=2)
        na_inter = (1 - self.alpha) * (target * (1 - pred)).sum(dim=2).sum(dim=2)
        loss = (1 - ((inter + 1.0) / (
            (1 + inter + a_inter + na_inter))))
        return loss.mean()

    def tfl(self, pred, weights, target):
        loss = self.tl(pred, weights, target)
        return torch.pow(loss, self.gamma)
