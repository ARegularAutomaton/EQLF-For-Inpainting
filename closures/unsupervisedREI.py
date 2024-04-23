import os
import torch
from torch.optim import Adam
import numpy as np
import random
from statistics import mean
from utils.log import Log
from models.unet import UNet
from models.odenet import ODENet
from datetime import datetime
from utils.storage import storage
from utils.metrics import cal_mse, cal_psnr, cal_ssim
from utils.lr import adjust_learning_rate

class REIUnsupervised():
    def __init__(self, in_channels=1, out_channels=1, net="unet", problem="inpaint", dtype=torch.float, device='cuda:0') -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = net
        self.problem = problem
        self.device = device
        self.dtype = dtype
        self.trained_model_path = None
        self.log_path = None

    def train(self, dataloader, operator, transform, epochs, lr=1e-4, wd=1e-8):
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

        filename = 'rei_{}_{}'.format(self.net, datetime.now().strftime('%d%m%Y'))
        log = Log(self.log_path, filename, ['epoch', 'loss_sure', 'loss_req', 'loss_total', 'psnr', 'mse', 'ssim', 'gpu_memory'])
        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr, False, epochs, [100,200,300,400])
            loss = self._closure(model, dataloader, operator, transform, optimizer, mse)
            log.record(epoch+1, *loss)
            self._print_stats(epoch, epochs, loss)
            self._save_model(epoch, model, optimizer)
        log.close()

    def _print_stats(self, epoch, epochs, loss):
        print('{}\tEpoch[{}/{}]\tsure={:.4e}\treq={:.4e}\tloss={:.4e}'.format(datetime.now().strftime('%y-%m-%d-%H:%M:%S'), epoch+1, epochs, *loss))
    
    def _save_model(self, epoch, model, optimizer):
        state = {'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(self.trained_model_path, 'rei_unsupervised_{}_{}.pth.tar'.format(self.net, datetime.now().strftime('%d%m%Y'))) )
         
    def _closure(self, net, dataloader, operator, transform, optimizer, criterion, tau=1e-2):
        loss_sure_seq, loss_req_seq, loss_seq, psnr_seq, mse_seq, ssim_seq, gpu_memory_seq = [], [], [], [], [], [], [] 
        for _, x in enumerate(dataloader):
            x = x[0] if isinstance(x, list) else x
            
            if self.problem == 'inpaint':
                inv = lambda y: net(operator.A_dagger(y))
                if len(x.shape) == 3:
                        x = x.unsqueeze()
                x = x.type(self.dtype).to(self.device)
                y0 = operator.A(x, add_noise=True)
                x0 = operator.A_dagger(y0)

                x1 = net(x0)
                y1 = operator.A(x1)

                # SURE-based unbiased estimator to the clean measurement consistency loss
                sigma2 = operator.noise_model['sigma'] ** 2
                b = torch.rand_like(x0)
                b = operator.A(b)
                y2 = operator.A(net(operator.A_dagger(y0 + tau * b))) # tau * b

                # req
                x2 = transform.apply(x1)
                x3 = inv(operator.A(x2, add_noise=True))
            
            if self.problem == 'ct':
                def to_device(x):
                    return x.unsqueeze(1).type(self.dtype).to(self.device) # torch operate on self.device
                def to_cpu(x):
                    return x.squeeze(1).detach().to('cpu') # astra toolbox operates on CPU
                
                MAX = 0.032/5
                MIN = 0
                I0 = 1e5
                norm = lambda x: (x-MIN) / (MAX-MIN)
                x = x * (MAX - MIN) + MIN
                log = lambda x: torch.log(torch.abs(I0 / x))

                f = lambda bp: net(norm(bp))
                A_dagger = lambda y, proj_id: to_device(operator.A_dagger(y, proj_id=proj_id, algorithm='FBP'))
                forwA = lambda x, angles: operator.A(to_cpu(x), add_noise=False, angles=angles)
                forwANoise = lambda x, angles: operator.A(to_cpu(x), add_noise=True, angles=angles)
                
                if len(x.shape)==4:
                    x = x.squeeze(1)
                
                angles = operator.get_angles()
                y0, proj_id = operator.A(x, add_noise=True, angles=angles)
                x0 = A_dagger(log(y0), proj_id)

                x1 = f(x0)
                y1, proj_id = forwA(x1, angles)

                # SURE-based unbiased estimator to the clean measurement consistency loss
                sigma2 = operator.noise_model['sigma'] ** 2
                b = torch.rand_like(x0)
                b, proj_id = forwA(b, angles)
                y2, proj_id = forwA(f(A_dagger(log(y0 + tau * b), proj_id)), angles)
                
                x2 = transform.apply(x1)
                m2, proj_id = forwANoise(x2, angles)
                x3 = f(A_dagger(log(m2), proj_id))

                x = x.type(self.dtype).to(self.device)
            
            # compute batch size K
            B = y0.shape[0]
            # compute n (dimension of x)
            n = y0.shape[-1]*y0.shape[-2]*y0.shape[-3]

            # compute m (dimension of y)
            if operator.name == 'inpainting':
                m = n * (1 - operator.mask_rate)
            elif operator.name == 'ct':
                m = n
            
            # compute sure loss
            loss_sure = ((torch.sum((y0 - y1).pow(2))) / (B * m)) - sigma2  \
                       + ((2 * sigma2 / (tau * m * B)) * ((b * (y2 - y1)).sum())) 
            
            if self.problem == 'inpaint':
                loss_req = criterion(x3, x2)
            if self.problem == 'ct':
                loss_req = criterion(norm(x3), norm(x2))

            # loss_mc = criterion(y1, y0)

            loss = loss_sure + loss_req

            loss_sure_seq.append(loss_sure.item())
            loss_req_seq.append(loss_req.item())
            loss_seq.append(loss.item())
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))
            ssim_seq.append(cal_ssim(x1,x.squeeze(0)))
            gpu_memory_seq.append(torch.cuda.memory_allocated(device=None))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_closure = [np.mean(loss_sure_seq), np.mean(loss_req_seq), np.mean(loss_seq), np.mean(psnr_seq), np.mean(mse_seq), np.mean(ssim_seq), mean(gpu_memory_seq)]
        return loss_closure
