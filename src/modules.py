import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as Transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.nn import init

import torch.nn as nn
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, decay_rate=0, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))
        
    def decay_step(self):
        self.sigma = max(self.sigma - self.decay_rate, 0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

    
def convolution_block(
    in_channels, 
    out_channels, 
    kernel_size=4,
    stride=2, 
    padding=1, 
    bias=False, 
    batch_norm=True, 
    dropout=0.5,
    sigma=0.1, 
    decay_rate=0
):
    '''
    Helper function that compiles a Conv2d, BatchNorm, and Dropout layers sequentially
    inputs:
        in_channels: int, number of in channels
        out_channels: int, number of out channels
        kernel_size: int, kernel size (default=4)
        stride: int, stride size (default=2)
        padding: int, pad size (default=1)
        bias: bool, adds learnable bias term (default=False)
        batch_norm: bool, adds a batch norm layer after each convolution layer (default=True)
        dropout: float, probability of dropout (default=0.25)
    '''
    layers = []

    # add noise
    layers.append(
        GaussianNoise(sigma=sigma, decay_rate=decay_rate)
    )
    
    # define convolution layer
    layers.append(
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            bias=bias
        )
    )
    
    if batch_norm:
        # define batch normalization layer
        layers.append(nn.BatchNorm2d(out_channels))

    if dropout:
        # add dropout
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim, sigma=0.1, decay_rate=0):
        '''
        Initialize the Discriminator Module
        inputs:
            conv_dim: integer, the depth of the first convolutional layer
        '''
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim
        self.sigma = sigma
        self.decay_rate = decay_rate

        # define first convolutional layer (BatchNorm2d = False)
        self.input = convolution_block(
            3, conv_dim, batch_norm=False, dropout=0)

        # define additional convolutional layers (BatchNorm2d = True)
        self.conv1 = convolution_block(conv_dim, conv_dim*2, sigma=sigma, decay_rate=decay_rate)
        self.conv2 = convolution_block(conv_dim*2, conv_dim*4, sigma=sigma, decay_rate=decay_rate)

        # define final fully connected layer
        self.linear_dim = conv_dim * 8 * 2
        self.classifier = nn.Sequential(*[
                GaussianNoise(),
                nn.Linear(self.linear_dim, 1)
            ]
        )                               

    def forward(self, x):
        '''
        Forward propagation of the neural network and returns the neural network's logits
        inputs:
            x: The input to the neural network
        '''
        # convolutional layer (BatchNorm2d = False)
        output = F.leaky_relu(self.input(x), 0.2, inplace=True)

        # convolutional layers (BatchNorm2d = True)
        output = F.leaky_relu(self.conv1(output), 0.2, inplace=True)
        output = F.leaky_relu(self.conv2(output), 0.2, inplace=True)

        # flatten
        output = output.view(-1, self.linear_dim)

        # final output layer
        output = self.classifier(output)
        return output


def deconvolution_block(in_channels, out_channels, kernel_size=4,
                        stride=2, padding=1, bias=False, batch_norm=True):
    '''
    Helper function that compiles a ConvTranspose2d, and BatchNorm layers sequentially
    inputs:
            in_channels: int, number of in channels
            out_channels: int, number of out channels
            kernel_size: int, kernel size (default=4)
            stride: int, stride size (default=2)
            padding: int, pad size (default=1)
            bias: bool, adds learnable bias term (default=False)
            batch_norm: bool, adds a batch norm layer after each convolution layer (default=True)
    '''
    layers = []

    # define deconvolution layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=bias))

    # define batch normalization layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    # add activation layer
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        '''
        Initialize the Generator Module
        inputs
            z_size: The length of the input latent vector, z
            conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        '''
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        # dense layer with output dimensions as the discriminatoir's last convolution layer
        self.input = nn.Linear(z_size, conv_dim * 4 * 2 * 2)

        # define additional convolutional layers (BatchNorm2d = True)
        self.deconv1 = deconvolution_block(conv_dim*16, conv_dim*8)
        self.deconv2 = deconvolution_block(conv_dim*8, conv_dim*4)
        self.deconv3 = deconvolution_block(conv_dim*4, conv_dim*2)
        self.deconv4 = deconvolution_block(conv_dim*2, conv_dim)

        self.outputs = nn.ConvTranspose2d(conv_dim, 3, kernel_size=4,
                                          stride=2, padding=1, bias=False)

    def forward(self, x):
        '''
        Forward propagation of the neural network. Returns a Tensor image as output
        inputs:
            x: The input to the neural network     
        '''

        # define feedforward behavior
        output = self.input(x)
        output = output.view(-1, self.conv_dim*16, 4, 4)  # reshape

        # hidden transpose conv layers + relu
        output = self.deconv1(output)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)

        output = torch.tanh(self.outputs(output))
        return output


def init_weights_normal(m, init_gain=0.02):
    '''
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    inputs:
        m: A module or layer in a network    
    '''
    classname = m.__class__.__name__

    # intitalize convolution for convolution and linear layers
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        # set weights to distribute normally around 0 with std of 0.02
        init.normal_(m.weight.data, 0.0, init_gain)

        # set the bias to 0 if the layer class has bias
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    # initalize batch normalization layers
    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights_xavier(m, init_gain=0.02):
    '''
    Applies initial weights to certain layers in a model .
    The weights are taken from a Xavier distribution 
    with mean = 0, std dev = 0.02.
    inputs:
        m: A module or layer in a network    
    '''
    classname = m.__class__.__name__

    # intitalize convolution for convolution and linear layers
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        # set weights to distribute normally around 0 with std of 0.02
        init.xavier_normal_(m.weight.data, gain=init_gain)

        # set the bias to 0 if the layer class has bias
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    # initalize batch normalization layers
    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
