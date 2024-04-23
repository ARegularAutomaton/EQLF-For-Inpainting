import os
from threading import local
import torch
import random
from datasets.dataset import data_enumerator
import matplotlib.pyplot as plt
from skimage import restoration
import numpy as np

class Inpainting():
    def __init__(self, img_dim=(256,256), localised="false", regions=10, mask_rate=0.5, noise_model=None ,device='cuda:0'):
        # img_dim in format (w,h) so the matrix of the image has dimension h*w.
        self.name = 'inpainting'
        self.localised = localised
        self.device = device
        self.regions = regions
        self.img_dim = img_dim
        self.mask_rate = mask_rate
        self.noise_model = noise_model if noise_model != None else {'noise_type':'g', 'sigma':0.1} # Gaussian
        self.mask = torch.ones(self.img_dim[1], self.img_dim[0]).to(self.device)

        if localised == "true":
            self.localised_masking()
        else:
            self.distributed_masking()

    def noise(self, x):
        y = x
        if self.noise_model is not None:
            # u = x
            # z = torch.poisson(abs(u / self.noise_model['gamma']))
            # y = self.noise_model['gamma'] * z
            y = y + torch.randn_like(x) * self.noise_model['sigma']
            y = torch.einsum('kl,ijkl->ijkl', self.mask, y)
        return y
    
    def localised_masking(self):
        for i in range(self.regions):
            x = random.randint(0,self.img_dim[0]-1)
            y = random.randint(0,self.img_dim[1]-1)

            x_patch = 50
            x_patch = random.randint(1,x_patch)
            y_patch = 50
            y_patch = random.randint(1,y_patch)
            self._mask_local(x,y,x_patch,y_patch)
    
    def _mask_local(self,x,y,x_patch,y_patch):
        for i in range(x_patch):
            for j in range(y_patch):
                px = min(x+i,self.img_dim[0]-1)
                py = min(y+j,self.img_dim[1]-1)
                self.mask[py][px] = 0

    def distributed_masking(self):
        mask_path = './operators/mask_random_{}x{}_{}.pt'.format(self.img_dim[0], self.img_dim[1], self.mask_rate)
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path).to(self.device)
        else:
            self.mask[torch.rand_like(self.mask) > 1 - self.mask_rate] = 0
            torch.save(self.mask, mask_path)
    
    def A(self, x, add_noise=False):
        y = self.noise(x) if add_noise else x
        return torch.einsum('kl,ijkl->ijkl', self.mask, y)

    def A_dagger(self, x):
        # return torch.einsum('kl,ijkl->ijkl', self.mask, x)
        x = x.squeeze() if len(x.shape) > 3 else x
        mask = torch.ones(self.img_dim[1], self.img_dim[0]).to(self.device)
        mask = mask - self.mask
        x = restoration.inpaint_biharmonic(x.detach().cpu().numpy(), mask.detach().cpu().numpy(), channel_axis=0)
        x = torch.from_numpy(x).to(self.device)
        x = x.unsqueeze(0)
        return x    

def toimage(x, problem='inpaint'):
        if problem == 'inpaint':
            return x.squeeze().detach().permute(1,2,0).cpu().numpy()
        elif problem == 'ct':
            return x.squeeze().detach().cpu().numpy()

if __name__ == "__main__":
    test_images = data_enumerator(train=False, batchsize=1, problem="inpaint")
    localised = Inpainting(img_dim=(256,256), localised="true")
    nonlocalised = Inpainting(img_dim=(256,256), localised="false")
    device = f"cuda:{0}"
    for i, x in enumerate(test_images):
        x = x[0] if isinstance(x, list) else x # batch_size = 1 is always a list of length two
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(torch.float).to(device)
        y_0 = localised.A(x, add_noise=False)
        y_1 = nonlocalised.A(x, add_noise=True)
        x = nonlocalised.A_dagger(y_1)

        plt.subplot(1,3,1)
        plt.imshow(toimage(x))
        plt.title('Ground truth')

        plt.subplot(1,3,2)
        plt.imshow(toimage(y_0))
        plt.title('Forward (Localised)')

        plt.subplot(1,3,3)
        plt.imshow(toimage(y_1))
        plt.title('Forward (Distributed)')

        plt.show()

