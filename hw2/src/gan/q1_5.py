import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
    #first loss term
    loss_fake = discrim_fake.mean()
    
    #second loss term
    loss_real = -discrim_real.mean()
    
    #third loss term
    #compute gradients of output w.r.t input
    loss_grad = torch.autograd.grad(discrim_interp, interp,\
                                    torch.ones_like(discrim_interp,\
                                    device="cuda"),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,)[0] #BxCxHxW
    
    #reshape gradients
    loss_grad = loss_grad.view(loss_grad.size(0), -1) #Bx-1
    
    #compute norm
    loss_grad = torch.linalg.norm(loss_grad, dim=-1) #Bx1

    #deviation of norm from 1
    loss_grad = (loss_grad - 1).pow(2).mean() #1 => scalar
    
    loss = loss_fake + loss_real + lamb*loss_grad
    
    return loss

def compute_generator_loss(discrim_fake):
    """
    TODO 1.5.1: Implement WGAN-GP loss for generator.
    loss = - E[D(fake_data)]
    """    
    #compute loss
    loss = -discrim_fake.mean()
    
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
