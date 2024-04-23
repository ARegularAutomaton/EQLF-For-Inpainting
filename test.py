from operators.ct import CT
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datasets.dataset import data_enumerator
from models.unet import UNet
from models.odenet import ODENet
from operators.inpainting import Inpainting
from utils.storage import storage
from utils.metrics import cal_mse, cal_psnr, cal_ssim
import os

parser = argparse.ArgumentParser()

# compulsory
parser.add_argument('--problem', default='inpaint', type=str, help='[TASK] Inpaint or CT')
parser.add_argument('--epochs', default=5000, type=int, help='[EPOCH]')
parser.add_argument('--model', default="", type=str, help="[FILENAME] Filename of the model in \"./trained_models\"")
parser.add_argument('--localised', default="false", type=str, help="[INPAINT] Localised (true), Distributed (false)")

# level 1
parser.add_argument('--net', default="unet", type=str, help="[NETWORK]: UNET or ODENET")

# level 2
parser.add_argument('--device', default='gpu', type=str, help="[DEVICE]: CPU or GPU")
parser.add_argument('--gpu', default=0, type=int, help="[GPU ID]: ID")
parser.add_argument('--size', default=(256,256), type=tuple, help='[IMGAGE SIZE]: (w,h)')

def test():
    args = parser.parse_args()
    device = f"cuda:{args.gpu}" if args.device == 'gpu' else 'cpu'
    test_images = data_enumerator(train=False, batchsize=1, problem=args.problem, crop_size=args.size, shuffle=False)

    if args.problem == 'inpaint':
        # model_path = storage.trained_model_path(epochs=args.epochs, localised=args.localised) + args.model
        forw = Inpainting(img_dim=args.size, localised=args.localised)
        psnr_list_learnt = list()
        ssim_list_learnt = list()
        psnr_list_pseudo = list()
        ssim_list_pseudo = list()
        psnr_list_measure = list()
        ssim_list_measure = list()
        for model_path in get_files_in_directory(storage.trained_model_path(net=args.net, problem=args.problem, operator=forw, epochs=args.epochs)):
            print(model_path)
            for _, x in enumerate(test_images):
                x = x[0] if isinstance(x, list) else x # batch_size = 1: each item is a 2-item list
                
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                x = x.type(torch.float).to(device)
                y = forw.A(x, add_noise=False)
                pseudo_x = forw.A_dagger(y)
                learned_x = invert(pseudo_x,device=device,network=args.net, model_path=model_path)

                show(y.squeeze().permute(1,2,0), pseudo_x.squeeze().permute(1,2,0),
                      learned_x.squeeze().permute(1,2,0), x.squeeze().permute(1,2,0))

                psnr = cal_psnr(y, x)
                # mse = cal_mse(y, x)
                ssim = cal_ssim(y,x.squeeze(0))
                psnr_list_measure.append(psnr)
                ssim_list_measure.append(ssim)

                psnr = cal_psnr(pseudo_x, x)
                # mse = cal_mse(pseudo_x, x)
                ssim = cal_ssim(pseudo_x,x.squeeze(0))
                psnr_list_pseudo.append(psnr)
                ssim_list_pseudo.append(ssim)

                psnr = cal_psnr(learned_x, x)
                # mse = cal_mse(learned_x, x)
                ssim = cal_ssim(learned_x,x.squeeze(0))
                psnr_list_learnt.append(psnr)
                ssim_list_learnt.append(ssim)
            
            print("PSNR (y):", "{:.3f} +/- {:.3f}".format(np.mean(psnr_list_measure),np.std(psnr_list_measure)))
            print("SSIM (y):", "{:.3f} +/- {:.3f}".format(np.mean(ssim_list_measure), np.std(ssim_list_measure)))

            print("PSNR (Pseudo):", "{:.3f} +/- {:.3f}".format(np.mean(psnr_list_pseudo), np.std(psnr_list_pseudo)))
            print("SSIM (Pseudo):", "{:.3f} +/- {:.3f}".format(np.mean(ssim_list_pseudo), np.std(ssim_list_pseudo)))

            print("PSNR (Learnt):", "{:.3f} +/- {:.3f}".format(np.mean(psnr_list_learnt), np.std(psnr_list_learnt)))
            print("SSIM (Learnt):", "{:.3f} +/- {:.3f}".format(np.mean(ssim_list_learnt), np.std(ssim_list_learnt)))

    elif args.problem == 'ct':
        pass

def show(y, pseudo, x_net, x):
    plt.subplot(1,4,1)
    plt.axis('off')
    plt.imshow(y.detach().cpu().numpy())
    plt.title('A')
    plt.text(252, 35, f'PSNR: {get_display_metric(y, x)}\n SSIM: {get_display_metric(y, x, 0)}', color='white', fontsize=12, ha='right')
    
    plt.subplot(1,4,2)
    plt.axis('off')
    plt.imshow(pseudo.detach().cpu().numpy())
    plt.title('B')
    plt.text(252, 35, f'PSNR: {get_display_metric(pseudo, x)}\n SSIM: {get_display_metric(pseudo, x, 0)}', color='white', fontsize=12, ha='right')
    
    plt.subplot(1,4,3)
    plt.axis('off')
    plt.imshow(x_net.detach().cpu().numpy())
    plt.title('C')
    plt.text(252, 35, f'PSNR: {get_display_metric(x_net, x)}\n SSIM: {get_display_metric(x_net, x, 0)}', color='white', fontsize=12, ha='right')

    plt.subplot(1,4,4)
    plt.axis('off')
    plt.imshow(x.detach().cpu().numpy())
    plt.title('D')
    plt.text(252, 35, f'', color='white', fontsize=12, ha='right')
    plt.show()

def get_display_metric(bp, x, name='psnr'):
    # print(bp.shape, x.shape)
    if name == 'psnr':
        return np.around(cal_psnr(bp, x), decimals=2).astype(str)
    else:
        return np.around(cal_ssim(bp, x.squeeze(0), multichannel=True, channel_axis=-1), decimals=2).astype(str)

def toimage(x, problem='inpaint'):
    if problem == 'inpaint':
        return x.squeeze().detach().permute(1,2,0).cpu().numpy()
    elif problem == 'ct':
        return x.squeeze().detach().cpu().numpy()

def invert(x, network, device, model_path, problem='inpaint'):
    if problem == 'inpaint':
        if network == "unet":
            net = UNet(input_channels=3, output_channels=3)
        else:
            net = ODENet(input_channels=3, output_channels=3)
    elif problem == 'ct':
        if network == "unet":
            net = UNet(input_channels=1, output_channels=1)
        else:
            net = ODENet(input_channels=1, output_channels=1)
    trained_model = torch.load(model_path, map_location=device)
    net.load_state_dict(trained_model['net'])
    net.to(device).eval()
    return net.forward(x)

def get_files_in_directory(directory):
    files_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)

    return files_list

if __name__ == '__main__':
    test()
