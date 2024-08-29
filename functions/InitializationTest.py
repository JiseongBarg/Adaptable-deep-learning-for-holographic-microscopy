import argparse
from math import pi

def parse_args():
    
    parser = argparse.ArgumentParser()

    # gpu device
    parser.add_argument("--device",default= 0, type=int, help="number of device [0,1]")

    # data
    parser.add_argument("--data_root",default= './Datasets/test_fig5', type=str, help="Path to data folder")
    parser.add_argument("--result_root",default = './Test', type=str, help="Path to save folder")
    parser.add_argument("--experiment", default = 'MainFig5',type=str, help="experiment name")
    parser.add_argument("--pretrained", default = 'D:/Synology_JS/Mooo Workspace_JS/1. Research Project/2024_ongoing_Generalization of phase retrieval_JSBarg_CSLee/Pretrained',type=str, help="experiment name")
    

    # Parameters for network
    parser.add_argument("--norm_use", default=True, type=bool)
    parser.add_argument("--lrelu_use", default=True, type=bool)
    parser.add_argument("--lrelu_slope", default=0.1, type=float)
    parser.add_argument("--zero_padding", default=True, type=bool)
    parser.add_argument("--initial_channel_G", default=48, type=int)
    parser.add_argument("--output_channel", default = 2, type=int)
    parser.add_argument("--methods", default='proposed', type=str)

    # Parameters for test
    parser.add_argument("--crop_size", default = 256, type=int)

    # Configurations for training dataset 
    parser.add_argument("--wavelength", default=532e-9, type=float)
    parser.add_argument("--pixel_size", default=6.5e-6, type=float)
    parser.add_argument("--numerical_aperture", default = 0.4, type=int)
    parser.add_argument("--magnification", default = 300/9, type=int)  
    
    parser.add_argument("--phase_normalize", default = 2*pi, type=float)

    return parser.parse_args()