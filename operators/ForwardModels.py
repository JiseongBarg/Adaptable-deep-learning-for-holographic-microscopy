from torch import nn
import torch
from math import pi
import numpy as np
from functions.MyTools import center_crop
from torch.nn.functional import pad
from math import factorial


class EffectiveForward(nn.Module):
    '''
    Implementation for distance parameterized physical forward model.
    We use Angular Spectrum method as forward model and object-to-sensor distance is parameterized.
    '''

    def __init__(self, args):
        super(EffectiveForward, self).__init__()
        self.wavelength = args.wavelength
        self.zero_padding = args.zero_padding
        self.phase_normalize = args.phase_normalize
        self.crop_size = args.crop_size
        self.device = args.device
        self.batch = args.batch_size

        ddx = np.linspace(0,self.crop_size+2*(self.crop_size//2)-1,self.crop_size+2*(self.crop_size//2))
        ddy = np.linspace(0,self.crop_size+2*(self.crop_size//2)-1,self.crop_size+2*(self.crop_size//2))

        ddx = (ddx-np.fix((self.crop_size+2*(self.crop_size//2))/2)) / (self.crop_size+2*(self.crop_size//2))
        ddy = (ddy-np.fix((self.crop_size+2*(self.crop_size//2))/2)) / (self.crop_size+2*(self.crop_size//2))
        
        ddx ,ddy = np.meshgrid(ddx, ddy)
        self.shape = ddx.shape
        ddx = np.expand_dims(ddx,axis=(0,1))
        ddy = np.expand_dims(ddy,axis=(0,1))
        self.ddx = torch.from_numpy(np.repeat(ddx,self.batch,axis=0))
        self.ddy = torch.from_numpy(np.repeat(ddy,self.batch,axis=0))


    def forward(self,O_raw, deff, peff,return_O = False, back = False, require_grad = True):
        
        batch, c, Sh, Sw = O_raw.shape
        # peff = peff.unsqueeze(-1)
        # print(peff.shape)
      
        ddx = self.ddx.to(self.device)/peff
        ddy = self.ddy.to(self.device)/peff

        # print(ddx.shape)

        CTF = torch.zeros(self.ddx.shape)

        CTF[ddx**2+ddy**2 <= 1] = torch.ones(1)

        CTF = CTF[:batch].to(self.device).detach()

        O_raw = pad(O_raw, pad=(Sh//2, Sh//2, Sw//2, Sw//2), mode="replicate")

        fx = (torch.arange(Sh*2)/2 - Sh//2).reshape([1, Sh*2, 1]).repeat(batch, 1, 1)
        fy = (torch.arange(Sw*2)/2 - Sw//2).reshape([1, 1, Sw*2]).repeat(batch, 1, 1)

        
        # print(peff.shape[0])

        if peff.shape[0] < 2:
            peff = peff.repeat(batch, 1, 1)
        else:
            peff = peff.view(batch, 1, 1)

        fx = fx.to(self.device)/(Sh*peff)  # batch, Sh, 1
        fy = fy.to(self.device)/(Sw*peff)

        G_in = (torch.bmm(torch.ones(size=[batch, fx.shape[1], 1]).to(self.device), torch.transpose(fx**2, 2, 1))
                          + torch.transpose(torch.bmm(torch.ones(size=[batch, fy.shape[2], 1]).to(self.device), fy**2),2,1))
        
        G_in = ((G_in>0)*1*G_in).view(batch, 1, fx.shape[1], fy.shape[2])

        G_in = G_in[:batch].to(self.device)
      
        G_in.requires_grad_(require_grad)
        
        if back:
            G_in = torch.exp(-1j*pi*-deff*G_in)
        else:
            G_in = torch.exp(-1j*pi*deff*G_in)

        O_fft = self.torch_fft(O_raw)
        H = self.torch_ifft(O_fft*G_in*CTF)
        H = center_crop(H,Sh)   
        
        I = torch.pow(torch.abs(H), 2)

        if return_O:   

            return H
        
        return I
    
    def torch_fft(self,H):
        
        H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))

        return H

    def torch_ifft(self,H):

        H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))

        return H



class ConventionalForward(nn.Module):
    '''
    Implementation for distance parameterized physical forward model.
    We use Angular Spectrum method as forward model and object-to-sensor distance is parameterized.
    '''

    def __init__(self, args):
        super(ConventionalForward, self).__init__()
        self.wavelength = args.wavelength
        self.px = args.pixel_size
        self.zero_padding = args.zero_padding
        self.phase_normalize = args.phase_normalize
        self.crop_size = args.crop_size
        self.device = args.device
        self.batch = args.batch_size
        
        scale = args.wavelength*args.magnification / args.numerical_aperture
        scale = args.pixel_size / scale
        
        ddx = np.linspace(1,self.crop_size+2*(self.crop_size//2),self.crop_size+2*(self.crop_size//2))
        ddy = np.linspace(1,self.crop_size+2*(self.crop_size//2),self.crop_size+2*(self.crop_size//2))

        ddx = (ddx-np.fix((self.crop_size+2*(self.crop_size//2))/2)) / (self.crop_size+2*(self.crop_size//2))
        ddy = (ddy-np.fix((self.crop_size+2*(self.crop_size//2))/2)) / (self.crop_size+2*(self.crop_size//2))
        
        ddx ,ddy = np.meshgrid(ddx, ddy)
        
        ddx = np.expand_dims(ddx,axis=(0,1)) / scale
        ddy = np.expand_dims(ddy,axis=(0,1)) / scale
        self.ddx = torch.from_numpy(np.repeat(ddx,self.batch,axis=0))
        self.ddy = torch.from_numpy(np.repeat(ddy,self.batch,axis=0))
        
        self.CTF = torch.zeros(self.ddx.shape)
        self.CTF[self.ddx**2+self.ddy**2 <= 1] = torch.ones(1)


    def forward(self,O_raw, d, return_O = False,require_grad = True,back = False,NA_crop = True):


        batch, c, Sh, Sw = O_raw.shape

        CTF = self.CTF[:batch].to(self.device).detach()

        O_raw = pad(O_raw, pad=(Sh//2, Sh//2, Sw//2, Sw//2), mode="replicate")

        fx = (torch.arange(Sh*2)/2 - Sh//2).reshape([1, Sh*2, 1]).repeat(batch, 1, 1)
        fy = (torch.arange(Sw*2)/2 - Sw//2).reshape([1, 1, Sw*2]).repeat(batch, 1, 1)

        px = self.px
        fx = fx.to(self.device)/(Sh*px)  # batch, Sh, 1
        fy = fy.to(self.device)/(Sw*px)

        G_in = 1 - (self.wavelength**2)*(torch.bmm(torch.ones(size=[batch, fx.shape[1], 1]).to(self.device), torch.transpose(fx**2, 2, 1))
                          + torch.transpose(torch.bmm(torch.ones(size=[batch, fy.shape[2], 1]).to(self.device), fy**2),2,1))
        
        G_in = (torch.sqrt((G_in>0)*1*G_in)/self.wavelength).view(batch, 1, fx.shape[1], fy.shape[2])

        if back:
            d = -d*1e-3
        else:
            d = d*1e-3

        G_in = G_in[:batch].to(self.device)
      
        G_in.requires_grad_(require_grad)
        G_in = torch.exp(1j*2*pi*d*G_in)

        O_fft = self.torch_fft(O_raw)
        if NA_crop:
            H = self.torch_ifft(O_fft*G_in*CTF)
        else:
            H = self.torch_ifft(O_fft*G_in)
        
        H = center_crop(H,Sh)   

        I = torch.pow(torch.abs(H), 2)

        if return_O:
            
            if NA_crop:
                O_NA = self.torch_ifft(O_fft*G_in*CTF)
                O_NA = center_crop(O_NA,Sh)        
                amp = torch.abs(O_NA)
                pha = torch.angle(O_NA)/self.phase_normalize
            else:
                O_NA = self.torch_ifft(O_fft*G_in)
                O_NA = center_crop(O_NA,Sh)        
                amp = torch.abs(O_NA)
                pha = torch.angle(O_NA)/self.phase_normalize

            return amp,pha
          
        return I
    
    def torch_fft(self,H):
        
        H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))

        return H

    def torch_ifft(self,H):

        H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))

        return H
    
