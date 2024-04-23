import os
import torch
import numpy as np
import argparse
from datasets.dataset import data_enumerator
from operators.inpainting import Inpainting
from operators.ct import CT
from transforms.shift import Shift
from transforms.rotate import Rotate
from transforms.permute import Permute
from closures.supervised import Supervised
from closures.supervisedEI import EISupervised
from closures.supervisedNoise import SupervisedNoisy
from closures.unsupervisedEI import EIUnsupervised
from closures.unsupervisedEIAdv import EIAdverserial
from closures.unsupervisedMC import MCUnsupervised
from closures.unsupervisedREI import REIUnsupervised

img_dim = (256,256) # Note: masking (forward operator) for the inpainting problem is 256x256

parser = argparse.ArgumentParser()

# compulsory
parser.add_argument('--problem', default="inpaint", type=str, help="[TASK] inpaint, ct")
parser.add_argument('--localised', default="false", type=str, help="[INPAINT] localised (T) or distributed (F) (default: false)")
parser.add_argument('--ct_type', default=1, type=str, help=f"[CT ARTIFACT] 1-4: Sparse-view, Limited-angle, Motion, Metal (default: {1})")
parser.add_argument('--closure', default=1, type=int, help=f"[OPTIMIZATION ALGORITHM] 1-7 (default: {1})")
parser.add_argument('--epochs', default=100, type=int, help=f"[EPOCHS] (default: {100})")
parser.add_argument('--transform', default=1, type=int, help=f"[EPOCHS] (default: {100})")

# level 1
parser.add_argument('--batch', default=1, type=int, help=f"[BATCH SIZE] (default: {1})")
parser.add_argument('--size', default=(256,256), type=tuple, help="[IMGAE SIZE] (w,h)")

# level 2
parser.add_argument('--net', default="unet", type=str, help="[ARCHITECTURE] UNET or ODENET")
parser.add_argument('--device', default='gpu', type=str, help="[CPU or GPU]")
parser.add_argument('--gpu', default=0, type=int, help="[GPU ID]")

# level 3
parser.add_argument('--lr', default=1e-4, type=float, help=f"[LEARNING RATE] (default: {1e-4})")
parser.add_argument('--aug', default=10, type=int, help="[ODENET] number of channels to augment")
parser.add_argument('--time_dependent', default=False, type=bool, help="[ODENET] whether the ode layers are time-dependent")
parser.add_argument('--back_prop', default="true", type=str, help="[ODENET] whether to use backpropagation")

def train():
    """
    NOTE: Only algorithms (closures) 1, 2, 6 has CT implementations (using astra-toolbox), but can only propagate EQ loss.
    Use the training script in repository REI-CT for training models for CT tasks.
    """

    # torch.cuda.memory._record_memory_history()
    args = parser.parse_args()
    device = f"cuda" if args.device == 'gpu' else 'cpu'
    noise_model = {'noise_type':'g', 'sigma':0.1}
    
    """
    Parameters
    """
    data_loader = data_enumerator(batchsize=args.batch, train=True, crop_size=args.size, problem=args.problem)
    inpainter = Inpainting(img_dim=args.size, localised=args.localised, device=device, noise_model=noise_model, regions=10) # regions: number of patches for localised inpainting
    ct = CT(type=args.ct_type)
    if args.transform == 1:
        transform = Permute(n_trans=1)
    elif args.transform == 2:
        transform = Shift(n_trans=1)
    elif args.transform == 3:
        transform = Rotate(n_trans=1)

    """
    EI Supervised (1)
    """
    if args.closure == 1:
        if args.problem == "inpaint":
            model = EISupervised(in_channels=3, out_channels=3, net=args.net, dtype=torch.float, problem=args.problem,
                                device=device)
            model.train(dataloader=data_loader, operator=inpainter, transform=transform, epochs=args.epochs, lr=args.lr)
        elif args.problem == "ct":
            model = EISupervised(in_channels=1, out_channels=1, net=args.net, dtype=torch.float, problem=args.problem,
                                device=device)
            model.train(dataloader=data_loader, operator=ct, transform=transform, epochs=args.epochs, lr=args.lr)

    """
    EI unpervised (2)
    """
    if args.closure == 2:
        if args.problem == "inpaint":
            model = EIUnsupervised(in_channels=3, out_channels=3, net=args.net, dtype=torch.float, problem=args.problem,
                            device=device)
            model.train(dataloader=data_loader, operator=inpainter, transform=transform, epochs=args.epochs, lr=args.lr)
        elif args.problem == "ct":
            model = EIUnsupervised(in_channels=1, out_channels=1, net=args.net, dtype=torch.float, problem=args.problem,
                            device=device)
            angles = np.linspace(0, np.pi, 180, endpoint=False)
            model.train(dataloader=data_loader, operator=ct, transform=transform, epochs=args.epochs, lr=args.lr)       

    """
    EI Adv (3)
    """    
    if args.closure == 3:
        if args.problem == "inpaint":
            model = EIAdverserial(in_channels=3, out_channels=3, net=args.net, dtype=torch.float, problem=args.problem,
                            device=device)
            model.train(dataloader=data_loader, operator=inpainter, transform=transform, epochs=args.epochs, lr=args.lr)

    """
    MC (4)
    """
    if args.closure == 4:
        if args.problem == "inpaint":
            model = MCUnsupervised(in_channels=3, out_channels=3, net=args.net, dtype=torch.float, problem=args.problem,
                                device=device)
            model.train(dataloader=data_loader, operator=inpainter, epochs=args.epochs, lr=args.lr)

    """
    REI unpervised (5)
    """      
    if args.closure == 5:
        if args.problem == "inpaint":
            model = REIUnsupervised(in_channels=3, out_channels=3, net=args.net, dtype=torch.float, problem=args.problem,
                                device=device)
            model.train(dataloader=data_loader, operator=inpainter, transform=transform, epochs=args.epochs, lr=args.lr)
        elif args.problem == "ct":
            model = REIUnsupervised(in_channels=1, out_channels=1, net=args.net, dtype=torch.float, problem=args.problem,
                                device=device)
            model.train(dataloader=data_loader, operator=ct, transform=transform, epochs=args.epochs, lr=args.lr)
    
    """
    Supervised noisy (6)
    """
    if args.closure == 6:
        model = SupervisedNoisy(in_channels=3, out_channels=3, net=args.net, dtype=torch.float, problem=args.problem,
                            device=device)
        model.train(dataloader=data_loader, operator=inpainter, epochs=args.epochs, lr=args.lr)
    
    """
    Supervised (7)
    """
    if args.closure == 7:
        model = Supervised(in_channels=3, out_channels=3, net=args.net, dtype=torch.float, problem=args.problem,
                            device=device)
        model.train(dataloader=data_loader, operator=inpainter, epochs=args.epochs, lr=args.lr)
    
    # torch.cuda.memory._dump_snapshot("memory")

if __name__ == '__main__':
    train()