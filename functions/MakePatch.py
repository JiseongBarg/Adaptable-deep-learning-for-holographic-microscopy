import torch
from torch import nn
import numpy as np

class make_patch(nn.Module):
    
    def __init__(self,args, size_org, size_new,spacing,batch_size):
        super(make_patch, self).__init__()

        self.device = args.device
        self.size_org = size_org
        self.size_new = size_new
        self.spacing = spacing
        self.batch_size = batch_size
        self.niter = ((self.size_org - self.size_new) // self.spacing) 
        self.nbatch = (self.niter**2 // self.batch_size)
    
    def forward(self,x):
        
        patched_x = torch.zeros([self.nbatch,self.batch_size,1,self.size_new,self.size_new]).to(self.device)

        id_stack = 0
        id_batch = 0
        # print(self.niter)

        for kk in range(self.niter):
            for jj in range(self.niter):
                temp_x = x[:,:, self.spacing*kk : self.size_new+ self.spacing*(kk) , self.spacing*jj : self.size_new+ self.spacing*(jj)]
                # print(temp_x.shape)
                patched_x[id_stack,id_batch,0,:,:] = temp_x

                id_batch += 1

                if id_batch == self.batch_size:
                    id_stack +=1
                    id_batch = 0

        return patched_x, self.nbatch
                








