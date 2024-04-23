import os
import torch
from torch.optim import Adam
import numpy as np
from statistics import mean
from utils.log import Log
from models.unet import UNet
from models.odenet import ODENet
from datetime import datetime
from utils.storage import storage
from utils.metrics import cal_mse, cal_psnr, cal_ssim


class EIUnsupervised():
    def __init__(self, in_channels=1, out_channels=1, net="unet", problem="inpaint", dtype=torch.float, device='cuda:0') -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = net
        self.problem = problem
        self.device = device
        self.dtype = dtype
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

        mc_criterion = torch.nn.MSELoss().to(self.device)
        ei_criterion = torch.nn.MSELoss().to(self.device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

        filename = 'ei_unsupervised_{}_{}_{}'.format(self.net, datetime.now().strftime('%d%m%Y'), transform.name)
        log = Log(self.log_path, filename, ['epoch', 'loss_mc', 'loss_ei', 'loss_total', 'psnr', 'mse', 'ssim', 'gpu_memory'])
        for epoch in range(epochs):
            loss = self._closure(model, dataloader, operator, transform,
                           optimizer, mc_criterion, ei_criterion)
            log.record(epoch+1, *loss)
            self._print_stats(epoch, epochs, loss)
            self._save_model(epoch, model, optimizer, filename)
        log.close()
    
    def _print_stats(self, epoch, epochs, loss):
        print('{}\tEpoch[{}/{}]\tmc={:.4e}\tei={:.4e}\tloss={:.4e}'.format(datetime.now().strftime('%y-%m-%d-%H:%M:%S'), epoch+1, epochs, *loss))
    
    def _save_model(self, epoch, model, optimizer, filename):
        state = {'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(self.trained_model_path, filename + '.pth.tar'))
         
    def _closure(self, net, dataloader, operator, transform, optimizer, mc, ei):
        loss_mc_seq, loss_ei_seq, loss_seq, psnr_seq, mse_seq, ssim_seq, gpu_memory_seq = [], [], [], [], [], [], []
        for _, x in enumerate(dataloader):
            x = x[0] if isinstance(x, list) else x # item are singletons of type tensor

            if self.problem == 'inpaint':
                if len(x.shape) == 3:
                        x = x.unsqueeze()
                x = x.type(self.dtype).to(self.device)

                y0 = operator.A(x)
                x0 = operator.A_dagger(y0)
                
                x1 = net(x0)
                y1 = operator.A(x1)

                x2 = transform.apply(x1)
                x3 = net(operator.A_dagger(operator.A(x2)))
            elif self.problem == 'ct':
                def to_device(x):
                    return x.unsqueeze(1).type(self.dtype).to(self.device) # torch operate on self.device
                def to_cpu(x):
                    return x.squeeze(1).detach().to('cpu') # astra toolbox operates on CPU
                if len(x.shape)==4:
                    x = x.squeeze(1)

                angles = operator.get_angles()

                y0, proj_id = operator.A(x, add_noise=True, angles=angles)
                x1 = net(to_device(operator.A_dagger(y0, proj_id=proj_id, algorithm='FBP')))
                y1, proj_id = operator.A(to_cpu(x1), angles=angles, add_noise=False)
                
                x2 = transform.apply(x1)
                tmp, proj_id = operator.A(to_cpu(x2), angles=angles, add_noise=False)
                x3 = net(to_device(operator.A_dagger(tmp, proj_id=proj_id, algorithm='FBP')))
                x = x.type(self.dtype).to(self.device)

            loss_mc = mc(y1,y0)
            loss_ei = ei(x3,x2)

            loss = loss_mc + loss_ei

            loss_mc_seq.append(loss_mc.item())
            loss_ei_seq.append(loss_ei.item())
            loss_seq.append(loss.item())
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))
            ssim_seq.append(cal_ssim(x1,x.squeeze(0)))
            gpu_memory_seq.append(torch.cuda.memory_allocated(device=None))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_closure = [np.mean(loss_mc_seq), np.mean(loss_ei_seq), np.mean(loss_seq), np.mean(psnr_seq), np.mean(mse_seq), np.mean(ssim_seq), mean(gpu_memory_seq)]

        return loss_closure
