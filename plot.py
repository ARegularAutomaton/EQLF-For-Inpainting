import argparse
from cProfile import label
from math import log
from operator import contains
import os
from threading import local
import numpy as np
import matplotlib.pyplot as plt
from utils.storage import storage
from operators.inpainting import Inpainting

parser = argparse.ArgumentParser()
parser.add_argument('--log', default="", type=str, 
                    help="name of csv file containing recorded losses in the logs directory")
parser.add_argument('--epochs', default=5000, type=int)
parser.add_argument('--localised', default="false", type=str, help="[INPAINT] Localised (true), Distributed (false)")

def plot():
    args = parser.parse_args()
    forw = Inpainting(img_dim=(256,256), localised=args.localised)
    path = storage.log_path(epochs=args.epochs, problem="inpaint", operator=forw)
    files = list_files(path)

    transforms = ['Permute', 'Rotation', 'Shift']
    headers = ["EPOCH","MC LOSS","EI LOSS","Loss","PSNR","MSE","SSIM","GPU Memory"]
    print(plt.style.available)
    plt.style.use('seaborn-v0_8')
    for f in range(len(files)):
        for c in range(4,7):
            arr = np.genfromtxt(files[f], delimiter=",", skip_header=1, usecols=(0,c))
            arr = arr.swapaxes(0,1)
            # print(arr)
            for i in range(len(arr[1])):
                arr[1][i] = arr[1][i] # log(abs(arr[1][i]))
            plt.plot(arr[0], arr[1], label=headers[c] + f" under {transforms[f]}")
    plt.xlabel("epochs")
    plt.ylabel("SSIM")
    plt.title("SSIM under different groups of transformations")
    plt.legend()
    plt.show()

def plot_all():
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.subplots_adjust(wspace=0.2)
    fig.suptitle('Graphs of loss and memory usage across epochs in the inpainting problem concerning distributed missing pixels')

    fig2, (ax3, ax4, ax5) = plt.subplots(1,3)
    fig2.subplots_adjust(wspace=0.2)
    fig2.suptitle('Graphs of reconstruction quality metrics across epochs in the inpainting problem concerning distributed missing pixels')
    forw = Inpainting(img_dim=(256,256), localised="false")
    path = storage.log_path(epochs=2000, problem="inpaint", operator=forw)

    ### 
    #Loss
    ###
    x = os.listdir(path)
    for fname in x:
        if 'adverserial' in fname or 'rei' in fname: continue
        # create label
        x=fname.split('_')
        label=''
        for s in x:
            if 'net' in s: break
            label += (s + ' ')
        # get the loss column
        names = np.genfromtxt(path + fname, delimiter=",", names=True).dtype.names
        for i in range(len(names)):
            if names[i] == "loss_mc" or names[i]=="loss_total" or names[i]=="loss_G" or names[i]=="loss":
                col = i
        # get data
        arr = np.genfromtxt(path + fname, delimiter=",", skip_header=1, usecols=(0,col))        
        arr = arr.swapaxes(0,1)
        ax1.plot(arr[0], [log(abs(x))/log(10) for x in arr[1]], label=label)
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.set_title('log-loss against epochs')

    ### 
    #GPU memory
    ###
    x = os.listdir(path)
    for fname in x:
        if 'adverserial' in fname or 'rei' in fname: continue
        # create label
        x=fname.split('_')
        label=''
        for s in x:
            if 'net' in s: break
            label += (s + ' ')
        # get the loss column
        names = np.genfromtxt(path + fname, delimiter=",", names=True).dtype.names
        for i in range(len(names)):
            if names[i] == "gpu_memory":
                col = i
        # get data
        arr = np.genfromtxt(path + fname, delimiter=",", skip_header=1, usecols=(0,col))        
        arr = arr.swapaxes(0,1)
        ax2.plot(arr[0], [log(abs(x))/log(10) for x in arr[1]], label=label)
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("memory")
    ax2.set_title('log-memory against epochs')

    
    ### 
    #MSE
    ###
    x = os.listdir(path)
    for fname in x:
        if 'adverserial' in fname or 'rei' in fname: continue
        # create label
        x=fname.split('_')
        label=''
        for s in x:
            if 'net' in s: break
            label += (s + ' ')
        # get the loss column
        names = np.genfromtxt(path + fname, delimiter=",", names=True).dtype.names
        for i in range(len(names)):
            if names[i] == "mse":
                col = i
        # get data
        arr = np.genfromtxt(path + fname, delimiter=",", skip_header=1, usecols=(0,col))        
        arr = arr.swapaxes(0,1)
        ax3.plot(arr[0], [log(abs(x))/log(10) for x in arr[1]], label=label)
    ax3.set_xlabel("epochs")
    ax3.set_ylabel("MSE")
    ax3.set_title('log-MSE against epochs')
    

    ### 
    #PSNR
    ###
    x = os.listdir(path)
    for fname in x:
        if 'adverserial' in fname or 'rei' in fname: continue
        # create label
        x=fname.split('_')
        label=''
        for s in x:
            if 'net' in s: break
            label += (s + ' ')
        # get the loss column
        names = np.genfromtxt(path + fname, delimiter=",", names=True).dtype.names
        for i in range(len(names)):
            if names[i] == "psnr":
                col = i
        # get data
        arr = np.genfromtxt(path + fname, delimiter=",", skip_header=1, usecols=(0,col))        
        arr = arr.swapaxes(0,1)
        ax4.plot(arr[0], [log(abs(x))/log(10) for x in arr[1]], label=label)
    ax4.set_xlabel("epochs")
    ax4.set_ylabel("PSNR")
    ax4.set_title('log-PSNR against epochs')

    ### 
    #SSIM
    ###
    x = os.listdir(path)
    for fname in x:
        if 'adverserial' in fname or 'rei' in fname: continue
        # create label
        x=fname.split('_')
        label=''
        for s in x:
            if 'net' in s: break
            label += (s + ' ')
        # get the loss column
        names = np.genfromtxt(path + fname, delimiter=",", names=True).dtype.names
        for i in range(len(names)):
            if names[i] == "ssim":
                col = i
        # get data
        arr = np.genfromtxt(path + fname, delimiter=",", skip_header=1, usecols=(0,col))        
        arr = arr.swapaxes(0,1)
        ax5.plot(arr[0], [log(abs(x))/log(10) for x in arr[1]], label=label)
    ax5.set_xlabel("epochs")
    ax5.set_ylabel("SSIM")
    ax5.set_title('log-SSIM against epochs')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    plt.show()


    # args = parser.parse_args()
    # arr = np.genfromtxt(storage.log_path() + "ei_adverserial_unsupervised_unet_16112023.csv", delimiter=",", skip_header=1, usecols=(0,4))
    # arr = arr.swapaxes(0,1)
    # plt.plot(arr[0], arr[1], label='ei adversarial (G)')

    # args = parser.parse_args()
    # arr = np.genfromtxt(storage.log_path() + "ei_adverserial_unsupervised_unet_16112023.csv", delimiter=",", skip_header=1, usecols=(0,5))
    # arr = arr.swapaxes(0,1)
    # plt.plot(arr[0], arr[1], label='ei adversarial (D)')

    # arr = np.genfromtxt(storage.log_path() + "ei_supervised_unet_16112023.csv", delimiter=",", skip_header=1, usecols=(0,3))
    # arr = arr.swapaxes(0,1)
    # plt.plot(arr[0], arr[1], label='ei supervised')

    # arr = np.genfromtxt(storage.log_path() + "ei_unsupervised_unet_16112023.csv", delimiter=",", skip_header=1, usecols=(0,3))
    # arr = arr.swapaxes(0,1)
    # plt.plot(arr[0], arr[1], label='ei unsupervised')

    # arr = np.genfromtxt(storage.log_path() + "mc_unsupervised_unet_16112023.csv", delimiter=",", skip_header=1, usecols=(0,1))
    # arr = arr.swapaxes(0,1)
    # plt.plot(arr[0], arr[1], label='mc unsupervised')

    # arr = np.genfromtxt(storage.log_path() + "rei_unsupervised_unet_16112023.csv", delimiter=",", skip_header=1, usecols=(0,3))
    # arr = arr.swapaxes(0,1)
    # plt.plot(arr[0], arr[1], label='rei unsupervised')

    # arr = np.genfromtxt(storage.log_path() + "supervised_unet_04122023.csv", delimiter=",", skip_header=1, usecols=(0,1))
    # arr = arr.swapaxes(0,1)
    # plt.subplot(1,2,2)
    # plt.plot(arr[0], arr[1], label='supervised noisy')
    # plt.set_xlabel("epochs")
    # plt.set_ylabel("loss")
    # plt.legend()


    # arr = np.genfromtxt(storage.log_path() + "supervised_noisy_unet_04122023.csv", delimiter=",", skip_header=1, usecols=(0,1))
    # arr = arr.swapaxes(0,1)
    # plt.subplot(1,2,1)
    # plt.plot(arr[0], arr[1], label='supervised')
    # plt.set_xlabel("epochs")
    # plt.set_ylabel("loss")
    # plt.legend()

def list_files(path):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

if __name__ == '__main__':
    # if args.log != "":
    plot()
    # else:
    #     plot_all()
