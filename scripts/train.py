from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional
from simple_parsing import Serializable, parse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

from tqdm import tqdm

from fastgan.models import weights_init, Discriminator, FastGAN
from fastgan.operation import copy_G_params, load_params
from fastgan.operation import ImageFolder, InfiniteSamplerWrapper
from fastgan.diffaug import DiffAugment
policy = 'color,translation'
from fastgan import lpips



def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

@dataclass   
class Trainer(Serializable):
    path: Path
    name: str = "test"
    total_iterations: int = 50000
    checkpoint: Optional[Path] = None
    batch_size: int = 8
    im_size: int = 1024
    ndf: int = 64
    ngf: int = 64
    nz: int = 256
    nlr: float = 0.0002
    nbeta1: float = 0.5
    dataloader_workers: int = 8
    current_iteration: int = 0
    save_interval: int = 100
    save_dir: Path = Path("./train_results")

    def train_d(self, data, label="real"):
        """Train function of discriminator"""
        if label=="real":
            pred, [rec_all, rec_small, rec_part], part = self.netD(data, label)
            err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
                self.percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
                self.percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
                self.percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = self.netD(data, label)
            err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()
        
    def train(self):
        task_name = self.save_dir / self.name
        while task_name.exists():
            response = input(f"Directory '{task_name}' already exists. Overwrite? (y/n): ").strip().lower()
            if response == 'y':
                shutil.rmtree(task_name)
                break
            elif response == 'n':
                new_name = input("Enter a new task name: ").strip()
                task_name = self.save_dir / new_name
            else:
                print("Please enter 'y' or 'n'.")

        saved_model_folder = task_name / 'models'
        saved_image_folder = task_name / 'images' 
        
        saved_model_folder.mkdir(parents=True)
        saved_image_folder.mkdir(parents=True)

        self.save(task_name / "config.yaml")

        device = torch.device("cpu")
        use_gpu = False
        if torch.cuda.is_available():
            device = torch.device("cuda")
            use_gpu = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

        transform_list = [
                transforms.Resize((int(self.im_size),int(self.im_size))),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        trans = transforms.Compose(transform_list)
        
        dataset = ImageFolder(root=self.path, transform=trans)

        dataloader = iter(DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                        sampler=InfiniteSamplerWrapper(dataset), num_workers=self.dataloader_workers, pin_memory=True))
        '''
        loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=dataloader_workers, 
                                pin_memory=True)
        dataloader = CudaDataLoader(loader, 'cuda')
        '''
        
        
        #from model_s import Generator, Discriminator
        netG = FastGAN(ngf=self.ngf, nz=self.nz, im_size=self.im_size)
        netG.apply(weights_init)

        self.netD = Discriminator(ndf=self.ndf, im_size=self.im_size)
        self.netD.apply(weights_init)

        self.percept = lpips.PerceptualLoss(model='net-lin', net='vgg')

        netG.to(device)
        self.netD.to(device)
        self.percept.to(device)

        avg_param_G = copy_G_params(netG)

        fixed_noise = netG.sample_latent(8).to(device)

        

        optimizerG = optim.Adam(netG.parameters(), lr=self.nlr, betas=(self.nbeta1, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.nlr, betas=(self.nbeta1, 0.999))
        
        if self.checkpoint is not None:
            ckpt = torch.load(self.checkpoint)
            netG.load_state_dict(ckpt['g'])
            self.netD.load_state_dict(ckpt['d'])
            avg_param_G = ckpt['g_ema']
            optimizerG.load_state_dict(ckpt['opt_g'])
            optimizerD.load_state_dict(ckpt['opt_d'])
            del ckpt

        for iteration in tqdm(range(self.current_iteration, self.total_iterations+1)):
            real_image = next(dataloader)
            real_image = real_image.to(device)
            current_batch_size = real_image.size(0)
            noise = netG.sample_latent(current_batch_size).to(device)

            fake_images = netG(noise)

            real_image = DiffAugment(real_image, policy=policy)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
            
            ## 2. train Discriminator
            self.netD.zero_grad()

            err_dr, rec_img_all, rec_img_small, rec_img_part = self.train_d(real_image, label="real")
            self.train_d([fi.detach() for fi in fake_images], label="fake")
            optimizerD.step()
            
            ## 3. train Generator
            netG.zero_grad()
            pred_g = self.netD(fake_images, "fake")
            err_g = -pred_g.mean()

            err_g.backward()
            optimizerG.step()

            for p, avg_p in zip(netG.parameters(), avg_param_G):
                avg_p.mul_(0.999).add_(0.001 * p.data)

            if iteration % 100 == 0:
                print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

            if iteration % (self.save_interval*10) == 0:
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)
                with torch.no_grad():
                    vutils.save_image(netG.generate(fixed_noise), saved_image_folder / f'{iteration:d}.jpg', nrow=4)
                    vutils.save_image( torch.cat([
                            F.interpolate(real_image, 128), 
                            rec_img_all, rec_img_small,
                            rec_img_part]).add(1).mul(0.5), saved_image_folder / f'rec_{iteration:d}.jpg' )
                load_params(netG, backup_para)

            if iteration % (self.save_interval*50) == 0 or iteration == self.total_iterations:
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)
                netG.save(saved_model_folder / f'{iteration:d}.pth')
                load_params(netG, backup_para)
                torch.save({'g':netG.state_dict(),
                            'd':self.netD.state_dict(),
                            'g_ema': avg_param_G,
                            'opt_g': optimizerG.state_dict(),
                            'opt_d': optimizerD.state_dict()}, saved_model_folder / f'all_{iteration:d}.pth')

if __name__ == "__main__":
    trainer = parse(Trainer)
    trainer.train()