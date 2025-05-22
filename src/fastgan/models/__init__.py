from enum import Enum
from pathlib import Path

import torch
from torch import nn

from .ffhq import FFHQDiscriminator
from .utils import InitLayer, SEBlock, UpBlock, UpBlockComp, conv2d
from .discriminator import DiscriminatorConfig
from .default import DefaultDiscriminator

class DiscriminatorEnum(Enum):
    DEFAULT = "default"
    FFHQ = "FFHQ"

def load_discriminator(config: DiscriminatorConfig = DiscriminatorConfig(), mode: DiscriminatorEnum = DiscriminatorEnum.DEFAULT):
    match mode:
        case DiscriminatorEnum.DEFAULT:
            return DefaultDiscriminator(config)
        case DiscriminatorEnum.FFHQ:
            return FFHQDiscriminator(config)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class FastGAN(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(FastGAN, self).__init__()

        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.init = InitLayer(nz, channel=nfc[4])
                                
        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])  
        self.feat_256 = UpBlock(nfc[128], nfc[256]) 

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False) 
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False) 
        
        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512]) 
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])  
        
    def forward(self, input):   
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)

        feat_64  = self.se_64( feat_4, self.feat_64(feat_32) )

        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )

        feat_256 = self.se_256( feat_16, self.feat_256(feat_128) )

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]
        
        feat_512 = self.se_512( feat_32, self.feat_512(feat_256) )
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]
    
    def sample_latent(self, num_samples):
        """Generate random latent vectors for the FastGAN model."""
        # Ensure num_samples is a positive integer
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        # Generate random latent vectors
        latents = torch.Tensor(num_samples, self.nz).normal_(0, 1)

        return latents

    def generate(self, input):
        """Generate images from the FastGAN model."""
        # Ensure the input is a tensor
        if not isinstance(input, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        # Check if the input size matches the expected size
        if input.size(1) != self.nz:
            raise ValueError(f"Input tensor must have {self.nz} channels, but got {input.size(1)}")

        # Forward pass through the model
        with torch.no_grad():
            generated_image = self.forward(input)[0]

        return generated_image.add(1).mul(0.5).clip(0, 1)

    def save(self, path: Path | str):
        """Save the FastGAN model to a given path."""
        # Convert path to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)

        # Check if the directory exists, if not create it
        if not path.parent.exists():
            path.parent.mkdir(parents=True)


        # Save the model
        torch.save({
            'model_state_dict': self.state_dict(),
            'im_size': self.im_size,
            'ngf': self.ngf,
            'nz': self.nz,
            'nc': self.nc,
            }, path)

    @staticmethod
    def load(path: Path | str):
        """Load the FastGAN model from a given path."""
        # Convert path to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)

        # Check if the path exists
        if not path.exists():
            raise FileNotFoundError(f"The path {path} does not exist.")

        # Load the model
        model_ckpt = torch.load(path, map_location='cpu')
        model = FastGAN(model_ckpt['ngf'], model_ckpt['nz'], model_ckpt['nc'], model_ckpt['im_size'])
        model.load_state_dict(model_ckpt['model_state_dict'])

        return model
