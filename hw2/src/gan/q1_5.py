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
    #apply sigmoid
    discrim_fake_prob = torch.sigmoid(discrim_fake)
    loss_fake = discrim_fake_prob.mean()
    
    #second loss term
    #apply sigmoid
    discrim_real_prob = torch.sigmoid(discrim_real)
    loss_real = -discrim_real_prob.mean()
    
    #third loss term
    loss_grad = torch.autograd.grad(interp, discrim_interp, retain_graph=True)
    loss_grad = (loss_grad).pow(2).mean()
    
    loss = loss_fake + loss_real + lamb*loss_grad
    
    return loss

def compute_generator_loss(discrim_fake):
    """
    TODO 1.5.1: Implement WGAN-GP loss for generator.
    loss = - E[D(fake_data)]
    """
    #apply sigmoid
    discrim_fake_prob = torch.sigmoid(discrim_fake)
    
    #compute loss
    loss = -discrim_fake_prob.mean()
    
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
