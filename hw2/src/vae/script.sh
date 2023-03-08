#python train.py --log_dir ae_latent16 --loss_mode ae --latent_size 16
#python train.py --log_dir ae_latent128 --loss_mode ae --latent_size 128
#python train.py --log_dir ae_latent1024 --loss_mode ae --latent_size 1024

#python train.py --log_dir vae_latent_rep --loss_mode vae --latent_size 1024

#python train.py --log_dir vae_latent_beta_.8 --loss_mode vae --latent_size 1024 --target_beta_val 0.8
#python train.py --log_dir vae_latent_beta_1.2 --loss_mode vae --latent_size 1024 --target_beta_val 1.2

python train.py --log_dir vae_latent_beta_annealing --loss_mode vae --latent_size 1024 --target_beta_val 0.8 --beta_mode linear