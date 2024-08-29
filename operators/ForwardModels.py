from torch import nn
import torch
from math import pi
import numpy as np
from functions.MyTools import center_crop
from torch.nn.functional import pad

class EffectiveForward(nn.Module):
    '''
    Implementation for physical forward model with effectivce parameterization.
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


    def forward(self,ObjectField, deff, peff, return_O = False, back = False, require_grad = True):
        
        batch, _, Sh, Sw = ObjectField.shape
        # peff = peff.unsqueeze(-1)
        # print(peff.shape)
      
        ddx = self.ddx.to(self.device)/peff
        ddy = self.ddy.to(self.device)/peff

        CTF = torch.zeros(self.ddx.shape)

        CTF[ddx**2+ddy**2 <= 1] = torch.ones(1)

        CTF = CTF[:batch].to(self.device).detach()

        ObjectField = pad(ObjectField, pad=(Sh//2, Sh//2, Sw//2, Sw//2), mode="replicate")

        fx = (torch.arange(Sh*2)/2 - Sh//2).reshape([1, Sh*2, 1]).repeat(batch, 1, 1)
        fy = (torch.arange(Sw*2)/2 - Sw//2).reshape([1, 1, Sw*2]).repeat(batch, 1, 1)

        if peff.shape[0] < 2:
            
            peff = peff.repeat(batch, 1, 1)
            
        else:
            
            peff = peff.view(batch, 1, 1)

        fx = fx.to(self.device)/(Sh*peff)
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

        O_fft = self.torch_fft(ObjectField)
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
    Implementation for physical forward model with only parameterized distance.

    '''

    def __init__(self, args):
        
        super(ConventionalForward, self).__init__()
        
        self.device = args.device
        self.lamb = args.wavelength
        self.pixel_size = args.pixel_size
        self.zero_padding = args.zero_padding
        self.phase_normalize = args.phase_normalize
        self.crop_size = args.crop_size
        self.batch = args.batch_size

        ddx = np.linspace(1,self.crop_size+2*(self.crop_size//2),self.crop_size+2*(self.crop_size//2))
        ddy = np.linspace(1,self.crop_size+2*(self.crop_size//2),self.crop_size+2*(self.crop_size//2))

        ddx = (ddx-np.fix((self.crop_size+2*(self.crop_size//2))/2)) / (self.crop_size+2*(self.crop_size//2))
        ddy = (ddy-np.fix((self.crop_size+2*(self.crop_size//2))/2)) / (self.crop_size+2*(self.crop_size//2))
        
        ddx, ddy = np.meshgrid(ddx, ddy)
        
        ddx = np.expand_dims(ddx,axis=(0,1)) * self.lamb * args.magnification / self.pixel_size / args.numerical_aperture
        ddy = np.expand_dims(ddy,axis=(0,1)) * self.lamb * args.magnification / self.pixel_size / args.numerical_aperture
        
        self.ddx = torch.from_numpy(np.repeat(ddx,self.batch,axis=0))
        self.ddy = torch.from_numpy(np.repeat(ddy,self.batch,axis=0))
        
        self.CTF = torch.zeros(self.ddx.shape)
        self.CTF[self.ddx**2 + self.ddy**2 <= 1] = torch.ones(1)


    def forward(self, ObjectField, d,  require_grad = True, back = False):

        batch, _, Sh, Sw = ObjectField.shape

        CTF = self.CTF[:batch].to(self.device).detach()

        ObjectField = pad(ObjectField, pad=(Sh//2, Sh//2, Sw//2, Sw//2), mode="replicate")

        fx = (torch.arange(Sh*2)/2 - Sh//2).reshape([1, Sh*2, 1]).repeat(batch, 1, 1)
        fy = (torch.arange(Sw*2)/2 - Sw//2).reshape([1, 1, Sw*2]).repeat(batch, 1, 1)
        
        fx = fx.to(self.device)/(Sh*self.pixel_size)
        fy = fy.to(self.device)/(Sw*self.pixel_size)

        G_in = 1 - (self.lamb**2)*(torch.bmm(torch.ones(size=[batch, fx.shape[1], 1]).to(self.device), torch.transpose(fx**2, 2, 1))
                          + torch.transpose(torch.bmm(torch.ones(size=[batch, fy.shape[2], 1]).to(self.device), fy**2),2,1))
        
        G_in = (torch.sqrt((G_in>0)*1*G_in)/self.lamb).view(batch, 1, fx.shape[1], fy.shape[2])

        if back:
            
            d = -d*1e-3
            
        else:
            
            d = d*1e-3

        G_in = G_in[:batch].to(self.device)
      
        G_in.requires_grad_(require_grad)
        G_in = torch.exp(1j*2*pi*d*G_in)

        O_fft = self.torch_fft(ObjectField)
        
        H = self.torch_ifft(O_fft*G_in*CTF)

        H = center_crop(H,Sh)   

        I = torch.pow(torch.abs(H), 2)

        return I
    
    def torch_fft(self,H):
        
        H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))

        return H

    def torch_ifft(self,H):

        H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))

        return H
    
