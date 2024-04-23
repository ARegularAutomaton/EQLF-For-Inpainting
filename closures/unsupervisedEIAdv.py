import os
import torch
from torch.optim import Adam
import numpy as np
from statistics import mean
from utils.log import Log
from models.unet import UNet
from models.odenet import ODENet
from models.discriminator import Discriminator
from datetime import datetime
from utils.storage import storage
from utils.metrics import cal_mse, cal_psnr, cal_ssim


class EIAdverserial():
    def __init__(self, in_channels=1, out_channels=1, net="unet", problem="inpaint",dtype=torch.float, device='cuda:0', img_dim=(256,256)) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = net
        self.problem = problem
        self.device = device
        self.dtype = dtype
        self.img_dim = img_dim
        self.trained_model_path = None
        self.log_path = None
        
    
    def train(self, dataloader, operator, transform, epochs, lr=0.0001, wd=1e-8):
        self.trained_model_path = storage.trained_model_path(self.net, self.problem, operator, epochs)
        self.log_path = storage.log_path(self.net, self.problem, operator, epochs)
        
        os.makedirs(self.log_path, exist_ok=True)

        os.makedirs(self.trained_model_path, exist_ok=True)

        if self.net == "unet":
            model = UNet(input_channels=self.in_channels,
                        output_channels=self.out_channels)
        else:
            model = ODENet(input_channels=self.in_channels,
                        output_channels=self.out_channels)

        model = model.to(self.device)

        discriminator = Discriminator((self.in_channels, self.img_dim[0], self.img_dim[1]),  device =self.device)

        mc_criterion = torch.nn.MSELoss().to(self.device)
        ei_criterion = torch.nn.MSELoss().to(self.device)

        criterion_D = torch.nn.MSELoss().to(self.device)

        optimizer_G = Adam(model.parameters(), lr=lr, weight_decay=wd)
        optimizer_D = Adam(discriminator.parameters(), lr=lr, weight_decay=wd)

        filename = 'ei_adverserial_unsupervised_{}_{}'.format(self.net, datetime.now().strftime('%d%m%Y'))
        log = Log(self.log_path, filename, ['epoch', 'loss_mc', 'loss_ei', 'loss_g', 'loss_G', 'loss_D', 'psnr', 'mse', 'ssim', 'gpu_memory'])
        for epoch in range(epochs):
            loss = self._closure(model, discriminator, dataloader, operator, transform,
                           optimizer_G, optimizer_D, mc_criterion, ei_criterion, criterion_D)
            log.record(epoch+1, *loss)
            self._print_stats(epoch, epochs, loss)
            self._save_model(epoch, model, discriminator, optimizer_G, optimizer_D)
        log.close()

    def _print_stats(self, epoch, epochs, loss):
        print('{}\tEpoch[{}/{}]\tmc={:.4e}\tei={:.4e}\tloss={:.4e}'.format(datetime.now().strftime('%y-%m-%d-%H:%M:%S'), epoch+1, epochs, *loss))
    
    def _save_model(self, epoch, model, discriminator, optimizer_G, optimizer_D):
        state = {'epoch': epoch,
                'net': model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()}
        torch.save(state, os.path.join(self.trained_model_path, 'ei_adverserial_unsupervised_{}_{}.pth.tar'.format(self.net, datetime.now().strftime('%d%m%Y'))))
         
    def _closure(self, net, discriminator, dataloader, operator, transform, optimizer_G, optimizer_D, mc, ei, criterion_D):
        loss_mc_seq, loss_ei_seq, loss_g_seq, loss_G_seq, loss_D_seq, psnr_seq, mse_seq, ssim_seq, gpu_memory_seq = [], [], [], [], [], [], [], [], []

        for i, x in enumerate(dataloader):
            x = x[0] if isinstance(x, list) else x
            if len(x.shape)==3:
                x = x.unsqueeze(1)
            x = x.type(self.dtype).to(self.device)

            # Measurements
            y0 = operator.A(x)

            # Model range inputs
            x0 = operator.A_dagger(y0)

            # Adversarial ground truths
            valid = torch.ones(x.shape[0], *discriminator.output_shape, requires_grad=False).type(self.dtype).to(self.device)
            valid_ei = torch.ones(x.shape[0]*transform.n_trans, *discriminator.output_shape, requires_grad=False).type(self.dtype).to(self.device)
            fake_ei = torch.zeros(x.shape[0]*transform.n_trans, *discriminator.output_shape, requires_grad=False).type(self.dtype).to(self.device)
            
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images from range input A^+y
            x1 = net(x0)
            y1 = operator.A(x1)

            # EI: x2, x3
            x2 = transform.apply(x1)
            x3 = net(operator.A_dagger(operator.A(x2)))

            # Loss measures generator's ability to measurement consistency and ei
            loss_mc = mc(y1, y0)
            loss_ei = ei(x3, x2)

            # Loss measures generator's ability to fool the discriminator
            loss_g = criterion_D(discriminator(x2), valid_ei)

            loss_G = loss_mc + 1 * loss_ei + 1 * loss_g

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion_D(discriminator(x1.detach()), valid)
            fake_loss = criterion_D(discriminator(x2.detach()), fake_ei)
            loss_D = 0.5 * 1 * (real_loss + fake_loss)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            loss_mc_seq.append(loss_mc.item())
            loss_ei_seq.append(loss_ei.item())
            loss_g_seq.append(loss_g.item())
            loss_G_seq.append(loss_G.item())# total loss for G
            loss_D_seq.append(loss_D.item())# total loss for D
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))
            ssim_seq.append(cal_ssim(x1,x.squeeze(0)))
            gpu_memory_seq.append(torch.cuda.memory_allocated(device=None))

        loss_closure = [np.mean(loss_mc_seq), np.mean(loss_ei_seq), np.mean(loss_g_seq),
            np.mean(loss_G_seq), np.mean(loss_D_seq), np.mean(psnr_seq), np.mean(mse_seq), np.mean(ssim_seq), mean(gpu_memory_seq)]

        return loss_closure
