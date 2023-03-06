import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        """
        TODO 2.1 : Fill in self.convs following the given architecture
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """

        #TODO 2.1: fill in self.fc, such that output dimension is self.latent_dim
        #MY IMPLEMENTATION
        #initialize conv layers
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        #initialize fc layers
        self.ds_size = 256*((input_shape[1]//8)**2)
        self.fc = nn.Linear(self.ds_size, self.latent_dim)

    def forward(self, x):
        #TODO 2.1 : forward pass through the network, output should be of dimension : self.latent_dim
        #MY IMPLEMENTATION
        x = self.convs(x)
        out = self.fc(x.view(-1, self.ds_size))
        return out
        
class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        #TODO 2.4: fill in self.fc, such that output dimension is 2*self.latent_dim
        #MY IMPLEMENTATION
        #number of elements in input
        self.ds_size = 256*((input_shape[1]//8)**2)
        
        #define encoder
        self.fc = nn.Linear(self.ds_size, latent_dim*2)

    def forward(self, x):
        #TODO 2.4: forward pass through the network.
        # should return a tuple of 2 tensors, mu and log_std
        #MY IMPLEMENTATION
        #pass through conv layeres
        x = self.convs(x)
        
        #pass through fc
        x = self.fc(x.view(-1, self.ds_size))
        
        #split output into mu and sigma
        mu = x[:,:self.latent_dim]
        log_std = x[:,self.latent_dim:]
        
        return mu, log_std


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        #TODO 2.1: fill in self.base_size
        #MY IMPLEMENTATION
        self.base_size = output_shape[1]//8
        self.ds_size = 256*((self.base_size)**2)
        
        """
        TODO 2.1 : Fill in self.deconvs following the given architecture
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        #MY IMPLEMENTATION
        #fc layer
        self.fc = nn.Linear(self.latent_dim, self.ds_size)
        
        #deconv layers
        self.deconvs =  nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )

    def forward(self, z):
        #TODO 2.1: forward pass through the network, first through self.fc, then self.deconvs.
        #MY IMPLEMENTATION
        #pass through fc
        x = self.fc(z)
        
        #reshape and pass through deconvs
        x = x.view(-1, 256, self.base_size, self.base_size)
        x = self.deconvs(x)
        
        return x

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    #NOTE: You don't need to implement a forward function for AEModel. For implementing the loss functions in train.py, call model.encoder and model.decoder directly.
