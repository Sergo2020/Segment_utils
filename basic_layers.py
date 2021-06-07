from torch import nn
import torch.nn.functional as F
import torch


class Conv_Layer(nn.Module):
    """
    Modified 2D convolutional layer torch.nn.Conv2d(). Includes: activations, pooling and normalization.


    Attributes
    ----------
    in_c : int
        Number of channels of an input tensor.
    out_c : int
        Number of channels of an output tensor.
    kernel : int
        Kernel size in pixels.
    stride : int
        Stride size in pixels.
    padding : int
        Number of zero-padding pixels for each size.
    dilation: str
        Size of dilation in pixels.
    bias : bool
        Bias flag. If True bias will be included in layer.
    activ : (None, str)
        Activation type. If None, no activation will be applied.
        Supported activation types: 'relu', 'leak', 'gelu', 'selu', 'sigmoid', 'softmax'.
    norm : (None, str)
        Normalization type. If None, no normalization will be applied.
        Supported normalization types: 'bn'
    pool : (None, str)
        Pooling type. If None, no pooling will be applied.
        Supported pooling types: 'max', 'avg'

    Methods
    -------
    forward (x: torch.Tensor)
        Forward pass for tensor 'x'.
    """

    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int,
                 padding: int = 0, dilation: int = 1, bias: bool = True, activ: (None, str) = None,
                 norm: (None, str) = None, pool: (None, str) = None) -> None:
        super(Conv_Layer, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv', nn.Conv2d(in_c, out_c, kernel_size=kernel,
                                               stride=stride, dilation=dilation, padding=padding, bias=bias))

        if activ == 'leak':
            activ = nn.LeakyReLU(inplace=True)
        elif activ == 'relu':
            activ = nn.ReLU(inplace=True)
        elif activ == 'pleak':
            activ = nn.PReLU()
        elif activ == 'gelu':
            activ = nn.GELU()
        elif activ == 'selu':
            activ = nn.SELU()
        elif activ == 'sigmoid':
            activ = nn.Sigmoid()
        elif activ == 'softmax':
            activ = nn.Softmax(dim=1)
        if norm == 'bn':
            norm = nn.BatchNorm2d(out_c)
        if pool == 'max':
            pool = nn.MaxPool2d(2, 2)
        elif pool == 'avg':
            pool = nn.AvgPool2d(2, 2)

        if not norm is None:
            self.conv.add_module('norm', norm)

        if not pool is None:
            self.conv.add_module('pool', pool)

        if not activ is None:
            self.conv.add_module('activ', activ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class DeConv_Layer(nn.Module):
    """
    Modified 2D deconvolutional layer torch.nn.ConvTranspose2d(). Includes: activations, pooling and normalization.


    Attributes
    ----------
    in_c : int
        Number of channels of an input tensor.
    out_c : int
        Number of channels of an output tensor.
    kernel : int
        Kernel size in pixels.
    stride : int
        Stride size in pixels.
    padding : int
        Number of zero-padding pixels for each size.
    bias : bool
        Bias flag. If True bias will be included in layer.
    activ : (None, str)
        Activation type. If None, no activation will be applied.
        Supported activation types: 'relu', 'leak', 'gelu', 'selu', 'sigmoid', 'softmax'.
    norm : (None, str)
        Normalization type. If None, no normalization will be applied.
        Supported normalization types: 'bn'
    pool : (None, str)
        Pooling type. If None, no pooling will be applied.
        Supported pooling types: 'max', 'avg'

    Methods
    -------
    forward (x: torch.Tensor)
        Forward pass for tensor 'x'.
    """

    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int,
                 padding: int = 0, bias: bool = True, activ: (None, str) = None,
                 norm: (None, str) = None, pool: (None, str) = None) -> None:
        super(DeConv_Layer, self).__init__()
        self.deconv = nn.Sequential()
        self.deconv.add_module('deconv', nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel,
                                                            stride=stride, padding=padding, bias=bias))

        if activ == 'leak':
            activ = nn.LeakyReLU(inplace=True)
        elif activ == 'relu':
            activ = nn.ReLU(inplace=True)
        elif activ == 'pleak':
            activ = nn.PReLU()
        elif activ == 'gelu':
            activ = nn.GELU()
        elif activ == 'selu':
            activ = nn.SELU()
        elif activ == 'sigmoid':
            activ = nn.Sigmoid()
        elif activ == 'softmax':
            activ = nn.Softmax(dim=1)
        if norm == 'bn':
            norm = nn.BatchNorm2d(out_c)
        if pool == 'max':
            pool = nn.MaxPool2d(2, 2)
        elif pool == 'avg':
            pool = nn.AvgPool2d(2, 2)

        if not norm is None:
            self.deconv.add_module('norm', norm)

        if not pool is None:
            self.deconv.add_module('pool', pool)

        if not activ is None:
            self.deconv.add_module('activ', activ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        return x


class Conv_Block(nn.Module):
    """
    Convolutional blocks which described in original U-Net paper (https://arxiv.org/abs/1505.04597).
    Consists of two convolutional layers (Conv_Layer) with kernel = 3, stride = 1,
    batch normalization and padding 1.
    Includes: activations, pooling and normalization.


    Attributes
    ----------
    in_c : int
        Number of channels of an input tensor.
    out_c : int
        Number of channels of an output tensor.
    activ : (None, str)
        Activation type. If None, no activation will be applied.
        Supported activation types: 'relu', 'leak', 'gelu', 'selu', 'sigmoid', 'softmax'.
    pool : (None, str)
        Pooling type. If None, no pooling will be applied.
        Supported pooling types: 'up_stride', 'up_bilinear', 'up_nearest', 'down_max', 'down_stride'.

    Methods
    -------
    forward (x: torch.Tensor)
        Forward pass for tensor 'x'. Returns output of a block and pooled version of output.
        If no pooling is defined return 0 and output.

        return: tuple[torch.Tensor, torch.Tensor]
    """
    def __init__(self, in_c : int, out_c : int, activ: (None, str)=None, pool: (None, str) = None) -> None:
        super(Conv_Block, self).__init__()
        self.c1 = Conv_Layer(in_c, out_c, 3, 1, activ=activ, norm='bn', padding=1)
        self.c2 = Conv_Layer(out_c, out_c, 3, 1, activ=activ, norm='bn', padding=1)

        if pool == 'up_stride':
            self.pool = DeConv_Layer(out_c, out_c, 2, 2, norm='bn')
        elif pool == 'up_bilinear':
            self.pool = nn.Upsample(scale_factor=2, mode=pool[3:], align_corners=True)
        elif pool == 'up_nearest':
            self.pool = nn.Upsample(scale_factor=2, mode=pool[3:], align_corners=True)
        elif pool == 'down_max':
            self.pool = nn.MaxPool2d(2, 2)
        elif pool == 'down_stride':
            self.c2 = Conv_Layer(out_c, out_c, 3, 2, activ=activ, norm='bn', padding=1)
            self.pool = None
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.c2(self.c1(x))

        if self.pool:
            return x, self.pool(x)
        else:
            return torch.Tensor(0), x


# -------- Functions ----------------------------------------------------------

def concat_curr(prev : torch.Tensor, curr : torch.Tensor) -> torch.Tensor:
    """
    Matches 4D tensor by height and width and concatinates them by channels.


    prev : torch.Tensor

    curr : torch.Tensor

    return: torch.Tensor
        Order is [prev, curr]
    """
    diffY = prev.size()[2] - curr.size()[2]
    diffX = prev.size()[3] - curr.size()[3]

    curr = F.pad(curr, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

    x = torch.cat([prev, curr], dim=1)
    return x
