import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        # TODO 3.1: compute the cumulative products for current and previous timesteps
        #MY IMPLEMENTATION
        self.alphas_cumprod = torch.cumprod(alphas, 0)
        self.alphas_cumprod_prev =  torch.zeros_like(self.alphas_cumprod)
        self.alphas_cumprod_prev[1:] = self.alphas_cumprod[0:-1]        
        
        # TODO 3.1: pre-compute values needed for forward process
        #MY IMPLEMENTATION
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1/torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = -torch.sqrt((1-self.alphas_cumprod)/self.alphas_cumprod)

        # TODO 3.1: compute the coefficients for the mean
        #MY IMPLEMENTATION
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_cumprod_prev)/(1-self.alphas_cumprod))*self.betas
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = torch.sqrt(alphas)*((1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod))

        # TODO 3.1: compute posterior variance
        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        #MY IMPLEMENTATION
        self.posterior_variance = (((1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod))*self.betas)
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # TODO 3.1: Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0
        # hint: can use extract function from utils.py
        #MY IMPLEMENTATION
        #extract time
        t_ = t[0]
        
        #mean
        posterior_mean = self.posterior_mean_coef1[t_]*x_0 + self.posterior_mean_coef2[t_]*x_t
        
        #variance
        posterior_variance = self.posterior_variance[t_]
        
        #log variance
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t_]
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        # TODO 3.1: given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py.
        # clamp x_0 to [-1, 1]
        #MY IMPLEMENTATION
        #predict noise
        #print(x_t.size(), x_t[0].min(), x_t[0].max(), torch.abs(x_t[0]).mean())
        #print(t[0])
        pred_noise = self.model(x_t, t)
        #print(pred_noise.size(), pred_noise.min(), pred_noise.max(), torch.abs(pred_noise[0]).mean())
        
        # if t[0] == 900:
        #     import sys
        #     sys.exit()
        
        #extract time
        t_ = t[0]
        
        #use extract function
        #pred_noise = extract(pred_noise, t, x_t.shape)
        
        #estimate x_0
        x_0 = x_t*self.x_0_pred_coef_1[t_] + \
                pred_noise*self.x_0_pred_coef_2[t_]
        
        #clamp x_0 
        x_0 = torch.clamp(x_0, -1, 1)
        
        #print(x_0.size(), x_0[0].min(), x_0[0].max(), torch.abs(x_0[0]).mean())
        #print('####')
        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # TODO 3.1: given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?
        #first estimate x_0
        pred_noise, x_0 = self.model_predictions(x, t)
        
        #estimate mu and variance
        mu, var, _ = self.get_posterior_parameters(x_0, x, t)
        
        #sample z
        z = torch.randn(mu.shape, device=mu.device)
        
        #estimate x_t-1
        pred_img = mu + torch.sqrt(var)*z
        
        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        # TODO 3.2: Generate a list of times to sample from.
        #MY IMPLEMANTATION
        sampled_timesteps = torch.linspace(total_timesteps-1, 0, \
                        sampling_timesteps+1, \
                        dtype=torch.int,\
                        device=self.device)
                        
        return sampled_timesteps
    
    def get_time_pairs(self, times):
        # TODO 3.2: Generate a list of adjacent time pairs to sample from.
        #MY IMPLEMANTATION
        t = copy.deepcopy(times)
        t_1 = copy.deepcopy(times)
        
        time_pairs = zip(t[0:-1], t_1[1:])
        
        return time_pairs

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # TODO 3.2: Compute the output image for a single step of the DDIM sampling process.
        #MY IMPLEMANTATION

        #batch times
        batched_times = torch.full((img.shape[0],), tau_i, device=self.device, dtype=torch.long)
        
        # predict x_0 and the additive noise for tau_i
        pred_noise, x_0 = self.model_predictions(img, batched_times)
        
        # extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}
        alpha_t_1 = self.alphas_cumprod[tau_isub1]
        alpha_t = self.alphas_cumprod[tau_i]
        
        # compute \sigma_{\tau_{i}}
        beta_t = ((1-alpha_t_1)/(1-alpha_t))*self.betas[tau_isub1]
        sigma_t = eta*beta_t
        
        # compute the coefficient of \epsilon_{\tau_{i}}
        eps_coeff = torch.sqrt(1-alpha_t_1-sigma_t)
        
        #compute x_0 coefficient
        x_0_coeff = torch.sqrt(alpha_t_1)
        
        #compute mu
        mu = x_0_coeff * x_0 + eps_coeff*pred_noise
        
        # sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        # HINT: use the reparameterization trick
        #sample z
        z = torch.randn(mu.shape, device=mu.device)
        
        #estimate x_t-1
        pred_img = mu + sigma_t*z

        return pred_img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        #TODO 3.3: fill out based on the sample function above
        #MY IMPLEMENTATION
        z = z.view(shape)
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        return sample_fn(shape, z)
        
