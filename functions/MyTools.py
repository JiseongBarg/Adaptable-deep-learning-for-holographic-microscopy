import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from math import pi, sqrt

def center_crop(H, size):
    if H.dim()>3:
        batch, channel, Nh, Nw = H.size()
        return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]
    elif H.dim()>2:
        channel, Nh, Nw = H.size()
        if size%2 ==1:
            size = size-1
            return H[ :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]
        else:
            return H[ :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]
    else:
        Nh, Nw = H.size()
        return H[(Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]


def center_crop_numpy(H, size):
    Nh = H.shape[0]
    Nw = H.shape[1]

    return H[(Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def amp_pha_generate(real, imag):
    field = real + 1j*imag
    amplitude = np.abs(field)
    phase = np.angle(field)

    return amplitude, phase

def make_path(path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)
        
        
def RI2AP(Real, Imag):
    
    field = Real + 1j*Imag
    Amplitude = torch.abs(field)
    Phase = torch.angle(field)
    
    return Amplitude, Phase



def savefig(save_path, results_data,args):
    
    real_diff,real_amp,real_pha,recon_amp,recon_pha,real_dist = results_data
    real_dist = real_dist.squeeze()
  
    fig2 = plt.figure(2, figsize=[15, 10])

    plt.subplot(2, 3, 1)
    plt.title(f'real hologram at {real_dist:.1f}mm')
    plt.imshow(real_diff, cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title('gt amplitude')
    plt.imshow(real_amp, cmap='gray',vmax = 1,vmin = 0)
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.title('gt phase')
    plt.imshow(real_pha, cmap='hot',vmax = 2.5, vmin = -1)
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.title('recon amplitude ')
    plt.imshow(recon_amp, cmap='gray',vmax = 1,vmin = 0)
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(2, 3, 6)
    plt.title('recon phase')
    plt.imshow(recon_pha, cmap='hot',vmax = 2.5,vmin = -1)
    plt.axis('off')
    plt.colorbar()
    
    fig2.suptitle(f'{args.experiment}' ,fontsize=20)

    fig2.savefig(save_path)
    plt.close(fig2)