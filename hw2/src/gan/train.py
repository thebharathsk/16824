from glob import glob
import os
import torch
from tqdm import tqdm
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset

def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    # NOTE: don't do anything fancy for 2, hint: the input image is between 0 and 1.
    
    #MY IMPLEMENTATION
    ds_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K iterations.
    # The learning rate for the generator should be decayed to 0 over 100K iterations.
    
    #MY IMPLEMENTATION
    #generator optimizer
    optim_generator = torch.optim.Adam(gen.parameters(), 0.0002, [0, 0.9])
    #discriminator optimizer
    optim_discriminator = torch.optim.Adam(disc.parameters(), 0.0002, [0, 0.9])
    
    #generator scheduler
    scheduler_generator = torch.optim.lr_scheduler.StepLR(optim_generator, step_size=200, gamma=0.9)
    #generator discriminator
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optim_discriminator, step_size=1000, gamma=0.9)
    
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
    amp_enabled=True,
):
    torch.backends.cudnn.benchmark = True # speed up training
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="../datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)

    scaler = torch.cuda.amp.GradScaler()

    iters = 0
    fids_list = []
    iters_list = []
    pbar = tqdm(total = num_iterations)
    while iters < num_iterations:
        for train_batch in train_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                train_batch = train_batch.cuda()
                ############################ UPDATE DISCRIMINATOR ######################################
                # TODO 1.2: compute generator, discriminator and interpolated outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.

                #MY IMPLEMENTATION
                #generator output
                gen_out = gen(n_samples=train_batch.size(0))
                
                #discriminator outputs
                #true data
                disc_out_real = disc(train_batch)
                #fake data
                disc_out_fake = disc(gen_out)
                
                #compute loss
                discriminator_loss = disc_loss_fn(disc_out_real, disc_out_fake)
                
                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.

            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # TODO 1.2: compute generator and discriminator output on generated data.
                    #MY IMPLEMENTATION
                    #generator output
                    gen_out = gen(n_samples=train_batch.size(0))
                    
                    #discriminator outputs
                    #fake data
                    disc_out_fake = disc(gen_out)
                    
                    #compute loss
                    generator_loss = gen_loss_fn(disc_out_fake)
                    
                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        #MY IMPLEMENTATION
                        #generator output
                        gen_out = gen(n_samples=train_batch.size(0))
                        
                        #scale output
                        generated_samples = gen_out*0.5 + 0.5
                        
                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    if os.environ.get('PYTORCH_JIT', 1):
                        torch.jit.save(torch.jit.script(gen), prefix + "/generator.pt")
                        torch.jit.save(torch.jit.script(disc), prefix + "/discriminator.pt")
                    else:
                        torch.save(gen, prefix + "/generator.pt")
                        torch.save(disc, prefix + "/discriminator.pt")
                    fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
            pbar.update(1)
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
