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

class SupervisedNoisy:
    def __init__(self, in_channels=1, out_channels=1, net="unet", problem="inpaint", dtype=torch.float, device='cuda:0') -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = net
        self.problem = problem
        self.device = device
        self.dtype = dtype
        self.trained_model_path = None
        self.log_path = None


    def train(self, dataloader, operator, epochs, lr=0.0001, wd=1e-8):
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
        mse = torch.nn.MSELoss().to(self.device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

        filename = 'supervised_noisy_{}_{}'.format(self.net, datetime.now().strftime('%d%m%Y'))
        log = Log(self.log_path, filename, ['epoch', 'loss_total', 'psnr', 'mse', 'ssim', 'gpu_memory'])
        for epoch in range(epochs):
            loss = self._closure(model, dataloader, operator,
                        optimizer, mse)
            log.record(epoch+1, *loss)
            self._print_stats(epoch, epochs, loss)
            self._save_model(epoch, model, optimizer)
        log.close()

    def _print_stats(self, epoch, epochs, loss):
        print('{}\tEpoch[{}/{}]\tloss={:.4e}'.format(datetime.now().strftime('%y-%m-%d-%H:%M:%S'), epoch+1, epochs, *loss))
    
    def _save_model(self, epoch, model, optimizer):
        state = {'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict()}
        
        torch.save(state, os.path.join(self.trained_model_path, 'supervised_noisy_{}_{}.pth.tar'.format(self.net, datetime.now().strftime('%d%m%Y'))) )

    def _closure(self, net, dataloader, operator, optimizer, mse):
        loss_seq,psnr_seq, mse_seq, ssim_seq, gpu_memory_seq = [], [], [], [], [] 

        for i, x in enumerate(dataloader):
            f = lambda y: net(operator.A_dagger(y))
            x = x[0] if isinstance(x, list) else x

            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            x = x.type(self.dtype).to(self.device)

            y0 = operator.A(x, add_noise=True)
            x1 = f(y0)

            loss = mse(x1, x)

            loss_seq.append(loss.item())
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))
            ssim_seq.append(cal_ssim(x1,x.squeeze(0)))
            gpu_memory_seq.append(torch.cuda.memory_allocated(device=None))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_closure = [np.mean(loss_seq), np.mean(psnr_seq), np.mean(mse_seq), np.mean(ssim_seq), mean(gpu_memory_seq)]

        return loss_closure