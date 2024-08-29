import torch
from torch.utils.data import Dataset
from typing import  Callable,  Optional
import os
import numpy as np

np.random.seed(777)
    
#############################################################

from glob import glob

class DatasetBarg(Dataset):
    
    def __init__(self,
                 root: str,
                 data_type,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
        
    ) -> None:

        self.image_set = image_set
        self.transform = transform
        self.data_type = data_type

        if self.data_type == 'inten':
            self.data_path = os.path.join(root, image_set, 'inten')
        else:
            self.data_path = os.path.join(root,image_set,'field')

        self.matdata_list = glob(os.path.join(self.data_path,'*.mat'))
        self.matdata_list = [os.path.basename(i) for i in self.matdata_list]

    def __len__(self) -> int:
        return len(self.matdata_list)

    def __getitem__(self, index: int):

        if self.data_type == 'field':
            
            data_field = self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['field']
            data_field = self.transform(data_field)
            data_amp = self.MM_norm(torch.abs(data_field))   
            data_pha = torch.angle(data_field)
            data_pha = data_pha - data_pha.mean()
            
            return data_amp, data_pha
        
        else:

            data_inten = self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['inten']
            distance =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['dist']                 
            pixel_size =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['pix']
            magnification =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['magn']
            na =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['NA']
            data_inten = self.MM_norm(self.transform(data_inten))
            distance = torch.Tensor(distance).float()
            pixel_size = torch.Tensor(pixel_size)
            magnification = torch.Tensor(magnification)
            na = torch.Tensor(na)

            return data_inten, distance, pixel_size, magnification, na
        

            
    def MM_norm(self, data):
        
        data_max = data.amax(dim=[1,2],keepdim=True)
        data_min = data.amin(dim=[1,2],keepdim=True)
        data_max = data_max.expand_as(data)
        data_min = data_min.expand_as(data)
        data_norm = (data - data_min) / (data_max - data_min)
        
        return data_norm
        

    def load_matfile(self, path):
        
        import scipy.io as sio
        data = sio.loadmat(path)
        
        return data

#################################################################
    

class test_paper(Dataset):
    
    def __init__(self,
                 data_type : str,
                 root: str,
                 tissue: str,
                 transform: Optional[Callable] = None,
                 return_parameter: bool=None,

                 
    ) -> None:
        self.tissue = tissue
        self.data_type = data_type
        self.transform = transform
        self.return_parameter = return_parameter
        
        if self.data_type == 'inten':
            self.data_path = os.path.join(root,self.tissue,'inten')
        else:
            self.data_path = os.path.join(root,self.tissue,'field')

        self.matdata_list = os.listdir(self.data_path)
      
            

    def __len__(self) -> int:
        return len(self.matdata_list)

    def __getitem__(self, index: int):

        if self.data_type == 'field':
            data_field = self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['field']

            if self.transform is not None:
                
                data_field = self.transform(data_field)
                
                data_amp = torch.abs(data_field)
                data_amp_max = data_amp.amax(dim=[1,2],keepdim=True)
                data_amp_min = data_amp.amin(dim=[1,2],keepdim=True)
                data_amp_max = data_amp_max.expand_as(data_amp)
                data_amp_min = data_amp_min.expand_as(data_amp)
                
                data_amp_norm = (data_amp - data_amp_min) / (data_amp_max - data_amp_min)
                
                data_pha = torch.angle(data_field)
                data_pha = data_pha - data_pha.mean()
                
            return data_amp_norm, data_pha
        
        else:

            if self.return_parameter:
                data_inten = self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['inten']
                distance =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['dist']                 
                pixel_size =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['pix']
                magn =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['magn']
                na =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['NA']
             
                if self.transform is not None:

                    data_inten = self.transform(data_inten)
                    data_inten_max = data_inten.amax(dim=[1,2],keepdim=True)
                    data_inten_min = data_inten.amin(dim=[1,2],keepdim=True)
                    data_inten_max = data_inten_max.expand_as(data_inten)   
                    data_inten_min = data_inten_min.expand_as(data_inten)
                    
                    data_inten_norm = (data_inten - data_inten_min) / (data_inten_max - data_inten_min)
                    distance = torch.Tensor(distance).float()
                    pixel_size = torch.Tensor(pixel_size)
                    magn = torch.Tensor(magn)
                    na = torch.Tensor(na)

                    return data_inten_norm, distance, pixel_size, magn, na
                           
            else:
                data_inten = self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['inten']
                distance =self.load_matfile(os.path.join(self.data_path,self.matdata_list[index]))['dist']        

                if self.transform is not None:
                    data_inten = self.transform(data_inten)

                    data_inten_max = data_inten.amax(dim=[1,2],keepdim=True)
                    data_inten_min = data_inten.amin(dim=[1,2],keepdim=True)
                    data_inten_max = data_inten_max.expand_as(data_inten)
                    data_inten_min = data_inten_min.expand_as(data_inten)
                    
                    data_inten_norm = (data_inten - data_inten_min) / (data_inten_max - data_inten_min)
                    distance = torch.Tensor(distance).float()
    
                    return data_inten_norm,distance
                    # return data_inten_norm
            

    def load_matfile(self, path):
        import scipy.io as sio
        data = sio.loadmat(path)
        return data




#