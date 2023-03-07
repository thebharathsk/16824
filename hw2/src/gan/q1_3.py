import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


# def compute_discriminator_loss(
#     discrim_real, discrim_fake, discrim_interp=None, interp=None, lamb=None
# ):
#     """
#     TODO 1.3.1: Implement GAN loss for discriminator.
#     Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
#     """
#     #for generated samples
#     #compute probability of real-ness
#     disc_prob_fake = torch.sigmoid(discrim_fake)
    
#     #compute discriminator loss
#     loss_fake = torch.log(1 - disc_prob_fake).mean()

#     #for true samples
#     #compute probability of real-ness
#     disc_prob_real = torch.sigmoid(discrim_real)
    
#     #compute discriminator loss
#     loss_real = torch.log(disc_prob_real).mean()
    
#     #optimize over negative of loss
#     return - loss_fake - loss_real


# def compute_generator_loss(discrim_fake):
#     """
#     TODO 1.3.1: Implement GAN loss for generator.
#     """
#     #compute probability of real-ness
#     disc_prob = torch.sigmoid(discrim_fake)
    
#     #compute generator loss
#     loss = torch.log(1 - disc_prob).mean()
    
#     return loss

def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp=None, interp=None, lamb=None
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    #for generated samples
    #compute probability of real-ness
    #disc_prob_fake = torch.sigmoid(discrim_fake)
    disc_targets_fake = torch.zeros_like(discrim_fake, device=discrim_fake.device)
    
    #compute discriminator loss
    loss_fake = F.binary_cross_entropy_with_logits(discrim_fake, disc_targets_fake)

    #for true samples
    #compute probability of real-ness
    #disc_prob_real = torch.sigmoid(discrim_real)
    disc_targets_real = torch.ones_like(discrim_real, device=discrim_real.device)
    
    #compute discriminator loss
    loss_real = F.binary_cross_entropy_with_logits(discrim_real, disc_targets_real)
    
    return loss_fake + loss_real


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """
    #compute probability of real-ness
    #disc_prob = torch.sigmoid(discrim_fake)
    disc_targets = torch.ones_like(discrim_fake, device=discrim_fake.device)
    
    #compute generator loss
    loss = F.binary_cross_entropy_with_logits(discrim_fake, disc_targets)
    
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
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
