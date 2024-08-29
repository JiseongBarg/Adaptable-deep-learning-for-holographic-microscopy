import torch.nn as nn
import torch
from torch.nn.functional import pad
from functions.MyTools import center_crop

class ScalerPeff(nn.Module):
    
    """
    Implementation for scaler which transforms the complex-valued field with peff_origin
    into the complex-valued field with peff_target
    """
    
    def __init__(self,args):
        super(ScalerPeff, self).__init__()
        
        self.img_size = args.crop_size
        self.device = args.device
        self.peff_origin = args.numerical_aperture*args.pixel_size/args.wavelength/args.magnification

    def forward(self, amp, pha, peff_target, rand_crop=True, fix_peff = False):
        
        amp = nn.functional.pad(amp, pad=(self.img_size//2, self.img_size//2,self.img_size//2, self.img_size//2),mode = 'reflect')
        pha = nn.functional.pad(pha, pad=(self.img_size//2, self.img_size//2,self.img_size//2, self.img_size//2),mode = 'reflect')
        
        field = amp*torch.exp(1j*pha)
        
        b,c,h,w = field.shape
        
        peff_ratio =  peff_target /self.peff_origin 
        
        # the ratio of new spectrum >1 -> new spectrum should be shrinked | <1 new spectrum should be expandeed
        amp_new = torch.zeros(b,c,self.img_size,self.img_size).to(self.device)
        pha_new = torch.zeros(b,c,self.img_size,self.img_size).to(self.device)
        peff = torch.zeros(b,1,1,1).to(self.device)
            
        for bb in range(b):
            
            fft_field = self.torch_fft(field[bb,0])

            if fix_peff:
                
                scaled_fft = fft_field
            
            else:
                
                if peff_ratio[bb] > 1:
                    
                    scale_fov = torch.round(h/peff_ratio[bb]/2)
                    scaled_fft = center_crop(fft_field,2*scale_fov.type(torch.int))

                elif peff_ratio[bb] <1 :
                    
                    scale_fov = torch.round(h/peff_ratio[bb])
                    scale_pad = torch.round(abs(h-scale_fov)/2).type(torch.int).item()
                    scaled_fft = pad(fft_field,pad = (scale_pad,scale_pad,scale_pad,scale_pad),mode = 'constant')

                else:
                    
                    scaled_fft = fft_field

            h1,_ = scaled_fft.shape
                        
            new_field = self.torch_ifft(scaled_fft)
                       
            peff[bb] = self.peff_origin*(h/h1)

            amp_scale = (h1/h)**2

            if rand_crop:  #random crop for data augmentation
                
                if h1 > self.img_size:
                    
                    rc_point = torch.randint(1,(h1-self.img_size),(1,))
                    amp_new[bb,0] = torch.abs(new_field[rc_point:rc_point+self.img_size,rc_point:rc_point+self.img_size])*amp_scale
                    pha_new[bb,0]= torch.angle(new_field[rc_point:rc_point+self.img_size,rc_point:rc_point+self.img_size])
                    
                else:
                    
                    amp_new[bb,0] = torch.abs(new_field)*amp_scale
                    pha_new[bb,0] = torch.angle(new_field)
            else: #center crop
                
                if h1 > self.img_size:
                    
                    amp_new[bb,0] = torch.abs(new_field[h1//2 - self.img_size//2 :h1//2+self.img_size//2 ,h1//2 - self.img_size//2:h1//2+self.img_size//2])*amp_scale
                    pha_new[bb,0] = torch.angle(new_field[h1//2 - self.img_size//2 :h1//2+self.img_size//2 ,h1//2 - self.img_size//2:h1//2+self.img_size//2])
                    
                else:
                    
                    amp_new[bb,0] = torch.abs(new_field)*amp_scale
                    pha_new[bb,0] = torch.angle(new_field)
          

        return amp_new*torch.exp(1j*pha_new), peff
    
    def torch_fft(self,H):
    
        H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))
        return H

    def torch_ifft(self,H):

        H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))

        return H
    
    