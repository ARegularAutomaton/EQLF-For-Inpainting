import torch.nn as nn
from anode.models import ODEBlock
from anode.conv_models import Conv2dTime

class ODENet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, concat=True, residual=False, dense=True, device="cuda:0"):
        super().__init__()
        self.name = 'odenet'
        self.concat = concat
        self.residual = residual
        self.dense = dense
        self.device = device
        
        augment_dim_1 = 0
        augment_dim_2 = 0
        augment_dim_3 = 0
        augment_dim_4 = 0
        augment_dim_5 = 0
        augment_dim_6 = 0
        augment_dim_7 = 0
        augment_dim_8 = 0
        augment_dim_9 = 0

        self.ode1 = ConvODENet(in_channels=input_channels, num_filters=64, time_dependent=True, augment_dim=augment_dim_1)
        self.ode2 = ConvODENet(in_channels=input_channels+augment_dim_1, num_filters=128, time_dependent=True, augment_dim=augment_dim_2)
        self.ode3 = ConvODENet(in_channels=input_channels+augment_dim_2, num_filters=256, time_dependent=True, augment_dim=augment_dim_3)
        self.ode4 = ConvODENet(in_channels=input_channels+augment_dim_3, num_filters=512, time_dependent=True, augment_dim=augment_dim_4)
        self.ode5 = ConvODENet(in_channels=input_channels+augment_dim_4, num_filters=1028, time_dependent=True, augment_dim=augment_dim_5)
        self.ode6 = ConvODENet(in_channels=input_channels+augment_dim_5, num_filters=512, time_dependent=True, augment_dim=augment_dim_6)
        self.ode7 = ConvODENet(in_channels=input_channels+augment_dim_6, num_filters=256, time_dependent=True, augment_dim=augment_dim_7)
        self.ode8 = ConvODENet(in_channels=input_channels+augment_dim_7, num_filters=128, time_dependent=True, augment_dim=augment_dim_8)
        self.ode9 = ConvODENet(in_channels=input_channels+augment_dim_8, num_filters=64, time_dependent=True, augment_dim=augment_dim_9)
        self.convout = nn.Conv2d(input_channels+augment_dim_9, output_channels, kernel_size=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        y = self.ode1(x)
        y = self.downsample(y)
        y = self.ode2(y)
        y = self.downsample(y)
        y = self.ode3(y)
        y = self.downsample(y)
        y = self.ode4(y)
        y = self.downsample(y)

        y = self.ode5(y)
        y = self.upsample(y)
        y = self.ode6(y)
        y = self.upsample(y)
        y = self.ode7(y)
        y = self.upsample(y)
        y = self.ode8(y)
        y = self.upsample(y)
        y = self.ode9(y)
        y = self.convout(y)
        return y

class ConvODEFunc(nn.Module):
    def __init__(self, device, in_channels=1, num_filters=1, augment_dim=0,
                 time_dependent=True, non_linearity='relu'):
        super(ConvODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.channels = in_channels
        self.channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = Conv2dTime(self.channels, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv2 = Conv2dTime(self.num_filters, self.channels,
                                    kernel_size=3, stride=1, padding=1)
            # self.conv3 = Conv2dTime(self.num_filters, self.channels,
            #                         kernel_size=1, stride=1, padding=0)
        else:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(self.channels, self.num_filters, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm2d(self.num_filters),
                        nn.ReLU(inplace=True),
            )
            # self.conv2 = nn.Sequential(
            #                         nn.Conv2d(self.num_filters, self.channels, kernel_size=3, padding=1, bias=True),
            #                         nn.BatchNorm2d(self.channels),
            #                         nn.ReLU(inplace=True),
            #                         # nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1, bias=True),
            #                         # nn.BatchNorm2d(self.num_filters),
            #                         # nn.ReLU(inplace=True)
            #                     )
            # # self.conv3 = nn.Sequential(
            # #                     nn.Conv2d(self.channels, self.channels,
            # #                                 kernel_size=1, stride=1, padding=0),
            # #                     nn.BatchNorm2d(self.channels),
            # #                     nn.ReLU(inplace=True),
            # #                 )

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            out = self.conv1(t, x)
            # out = self.non_linearity(out)
            out = self.conv2(t, out)
            # out = self.non_linearity(out)
            # out = self.conv3(t, out)
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            # out = self.conv3(out)
        return out

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up_conv(x)
   
class ConvODENet(nn.Module):
    def __init__(self, in_channels, num_filters,
                 augment_dim=0, time_dependent=True, non_linearity='relu',
                 tol=1e-3, adjoint=False, device="cuda:0"):
        super(ConvODENet, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ConvODEFunc(device, in_channels, num_filters, augment_dim,
                              time_dependent, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint)

    def forward(self, x):
        features = self.odeblock(x, eval_times=None) # return y(T) where T is the final time if set to None
        return features