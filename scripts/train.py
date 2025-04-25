import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from fastgan.models import weights_init, Discriminator, FastGAN
from fastgan.operation import copy_G_params, load_params, get_dir
from fastgan.operation import ImageFolder, InfiniteSamplerWrapper
from fastgan.diffaug import DiffAugment
policy = 'color,translation'
from fastgan import lpips

#torch.backends.cudnn.benchmark = True


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


class Trainer:
    def __init__(self, args):
        data_root = args.path
        self.total_iterations = args.iter
        checkpoint = args.ckpt
        batch_size = args.batch_size
        im_size = args.im_size
        ndf = 64
        ngf = 64
        nz = 256
        nlr = 0.0002
        nbeta1 = 0.5
        dataloader_workers = 8
        self.current_iteration = 0
        self.save_interval = args.save_interval
        self.saved_model_folder, self.saved_image_folder = get_dir(args)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "gpu"
        elif torch.mps.is_available():
            self.device = "mps"

        transform_list = [
                transforms.Resize((int(im_size),int(im_size))),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        trans = transforms.Compose(transform_list)

        if 'lmdb' in data_root:
            from fastgan.operation import MultiResolutionDataset
            dataset = MultiResolutionDataset(data_root, trans, 1024)
        else:
            dataset = ImageFolder(root=data_root, transform=trans)

        self.dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
        '''
        loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=dataloader_workers, 
                                    pin_memory=True)
        dataloader = CudaDataLoader(loader, 'cuda')
        '''

        #from model_s import Generator, Discriminator
        self.netG = FastGAN(ngf=ngf, nz=nz, im_size=im_size)
        self.netG.apply(weights_init)

        self.netD = Discriminator(ndf=ndf, im_size=im_size)
        self.netD.apply(weights_init)

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.avg_param_G = copy_G_params(self.netG)

        self.fixed_noise = self.netG.sample_latent(8).to(self.device)

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

        if checkpoint != 'None':
            ckpt = torch.load(checkpoint)
            self.netG.load_state_dict(ckpt['g'])
            self.netD.load_state_dict(ckpt['d'])
            self.avg_param_G = ckpt['g_ema']
            self.optimizerG.load_state_dict(ckpt['opt_g'])
            self.optimizerD.load_state_dict(ckpt['opt_d'])
            self.current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
            del ckpt

        self.percept = lpips.PerceptualLoss(model='net-lin', net='vgg', device=self.device)

    def train_d(self, net, data, label="real"):
        """Train function of discriminator"""
        if label=="real":
            pred, [rec_all, rec_small, rec_part], part = net(data, label)
            err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
                self.percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
                self.percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
                self.percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = net(data, label)
            err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()


    def train(self):
        for iteration in tqdm(range(self.current_iteration, self.total_iterations+1)):
            real_image = next(self.dataloader)
            real_image = real_image.to(self.device)
            current_batch_size = real_image.size(0)
            noise = self.netG.sample_latent(current_batch_size).to(self.device)

            fake_images = self.netG(noise)

            real_image = DiffAugment(real_image, policy=policy)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
            
            ## 2. train Discriminator
            self.netD.zero_grad()

            err_dr, rec_img_all, rec_img_small, rec_img_part = self.train_d(self.netD, real_image, label="real")
            self.train_d(self.netD, [fi.detach() for fi in fake_images], label="fake")
            self.optimizerD.step()
            
            ## 3. train Generator
            self.netG.zero_grad()
            pred_g = self.netD(fake_images, "fake")
            err_g = -pred_g.mean()

            err_g.backward()
            self.optimizerG.step()

            for p, avg_p in zip(self.netG.parameters(), self.avg_param_G):
                avg_p.mul_(0.999).add_(0.001 * p.data)

            if iteration % 100 == 0:
                print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

            if iteration % (self.save_interval*10) == 0:
                backup_para = copy_G_params(self.netG)
                load_params(self.netG, self.avg_param_G)
                with torch.no_grad():
                    vutils.save_image(self.netG.generate(self.fixed_noise), self.saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                    vutils.save_image( torch.cat([
                            F.interpolate(real_image, 128), 
                            rec_img_all, rec_img_small,
                            rec_img_part]).add(1).mul(0.5), self.saved_image_folder+'/rec_%d.jpg'%iteration )
                load_params(self.netG, backup_para)

            if iteration % (self.save_interval*50) == 0 or iteration == self.total_iterations:
                backup_para = copy_G_params(self.netG)
                load_params(self.netG, self.avg_param_G)
                self.netG.save(self.saved_model_folder+'/%d.pth'%iteration)
                load_params(self.netG, backup_para)
                torch.save({'g':self.netG.state_dict(),
                            'd':self.netD.state_dict(),
                            'g_ema': self.avg_param_G,
                            'opt_g': self.optimizerG.state_dict(),
                            'opt_d': self.optimizerD.state_dict()}, self.saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--save_interval', type=int, default=100, help='interval for saving model and image')


    args = parser.parse_args()
    print(args)

    trainer = Trainer(args)

    trainer.train()