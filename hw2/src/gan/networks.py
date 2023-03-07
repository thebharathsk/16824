import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers
        #MY IMPLEMENTATION
        #useful parameters
        self.upscale_factor = upscale_factor
        
        #pixel shuffle layer
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        
        #conv layer
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=padding)
        
    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel wise upscale_factor^2 times
        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        # 3. Apply convolution and return output
        
        #MY IMPLEMENTATION        
        #repeat along channel axis
        x = torch.repeat_interleave(x, self.upscale_factor**2, 1) #NxC*d*dxHxW
        
        #pixel shuffle
        x = self.pixel_shuffle(x) #NxCxH*dxW*d
        
        #apply convolution
        x = self.conv(x) #NxC_outxH*dxW*d
        
        return x

class DownSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers

        #MY IMPLEMENTATION
        self.downscale_ratio = downscale_ratio
        
        #pixel unshuffle
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_ratio)
        
        #convolution
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=padding)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        # 2. Then split channel wise into (downscale_factor^2xbatch x channel x height x width) images
        # 3. Average across dimension 0, apply convolution and return output

        #MY IMPLEMENTATION
        C = x.size(0)
         
        #unshuffle
        x = self.pixel_unshuffle(x) #NxCxH*dxW*d => NxC*d*dxHxW
        
        #reshape
        _,_,H,W = x.size()
        x = x.unsqueeze(2) #NxC*d*dx1xHxW
        x = x.view(-1, C, self.downscale_ratio**2, H, W) #NxCxd*dxHxW
        
        #mean across 3rd dimension
        x = torch.mean(x, dim=2) #NxCxHxW
        
        #convolution
        x = self.conv(x) #NxC_outxHxW
        
        return x

class ResBlockUp(torch.jit.ScriptModule):
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        # TODO 1.1: Impement Residual Block Upsampler.
        """
        ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
                (5): UpSampleConv2D(
                    (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                )
            )
            (upsample_residual): UpSampleConv2D(
                (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
            )
        """
        super(ResBlockUp, self).__init__()
        # TODO 1.1: Setup the network layers
        #MY IMPLEMENTATION
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            UpSampleConv2D(n_filters, kernel_size=kernel_size, n_filters=n_filters, upscale_factor=2, padding=1)
            )
        
        self.upsample_residual = UpSampleConv2D(input_channels, kernel_size=1, \
                                                n_filters=n_filters, upscale_factor=2,\
                                                padding=0)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        #MY IMPLEMENTATION
        
        #pass through layers
        x_1 = self.layers(x)
        
        #upsample input
        x_2 = self.upsample_residual(x)
        
        #make the residual connection
        out = x_1 + x_2
        
        return out

class ResBlockDown(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # TODO 1.1: Setup the network layers
        #MY IMPLEMENTATION
        #layers
        self.layers =  nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            DownSampleConv2D(n_filters, kernel_size=kernel_size, n_filters=n_filters, downscale_ratio=2, padding=1))
        
        #residual downsampling
        self.downsample_residual =  DownSampleConv2D(input_channels, kernel_size=1,\
                                                    n_filters=n_filters, downscale_ratio=2,\
                                                    padding=0)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
                #MY IMPLEMENTATION
        
        #MY IMPLEMENTATION
        #pass through layers
        x_1 = self.layers(x)
        
        #upsample input
        x_2 = self.upsample_residual(x)
        
        #make the residual connection
        out = x_1 + x_2
        
        return out

class ResBlock(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        # TODO 1.1: Setup the network layers
        #MY IMPLEMENTATION
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))
        )

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        #MY IMPLEMENTATION
        x_res = self.layers(x)
        
        return x + x_res

class Generator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        # TODO 1.1: Setup the network layers
        #MY IMPLEMENTATION
        self.starting_image_size = 4
        self.in_features = 128
        self.dense = nn.Linear(in_features=128, out_features=2048, bias=True)
        self.layers = nn.Sequential(ResBlockUp(128, 3, 128),
                                    ResBlockUp(128, 3, 128),
                                    ResBlockUp(128, 3, 128),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.Tanh())

    @torch.jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        #MY IMPLEMENTATION
        #pass through dense layer
        z_out = self.dense(z)
        
        #reshape latent vectors
        z_out = z_out.view(z.size(0), self.in_features, \
                            self.starting_image_size, \
                            self.starting_image_size)
        
        #upscaling
        out = self.layers(z_out)
        
        return out
                
    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        #MY IMPLEMENTATION
        #generate latent vector
        z = torch.randn([n_samples, self.in_features])
        
        #generate images
        out = self.forward_given_samples(z)
        
        return out

class Discriminator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO 1.1: Setup the network layers
        #MY IMPLEMENTATION
        self.layers = nn.Sequential(ResBlockDown(3, 3, 128),
                                    ResBlockDown(128, 3, 128),
                                    ResBlock(128, 3, 128),
                                    ResBlock(128, 3, 128),
                                    nn.ReLU()
                                    )
        self.dense = nn.Linear(in_features=128, out_features=1, bias=True)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to sum across the image dimensions after passing x through self.layers.
        #MY IMPLEMENTATION
        
        #batch size
        B = x.size(0)
        
        #downscale
        x = self.layers(x)
        
        #reshape
        x = x.view(B, -1)
        
        #regress 
        x = self.dense(x)
        
        return x