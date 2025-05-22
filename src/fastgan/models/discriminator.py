from dataclasses import dataclass
from torch import nn


@dataclass
class DiscriminatorConfig:
    ndf: int = 64
    nc: int = 3
    im_size: int = 512

class Discriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config