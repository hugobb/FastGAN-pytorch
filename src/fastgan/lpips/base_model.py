import os
import torch
import numpy as np
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.image_paths = None
        self.save_dir = None
        
    def name(self) -> str:
        return 'BaseModel'

    def optimize_parameters(self):
        pass

    def get_current_errors(self):
        return {}

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if self.save_dir is None:
            raise ValueError("self.save_dir is None, please set it before loading a network.")
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        if self.save_dir is None:
            raise ValueError("self.save_dir is None, please set it before saving done flag.")
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')

