import argparse
from math import pi

def parse_args():
    
    parser = argparse.ArgumentParser()

    # gpu device
    parser.add_argument("--device",default= 0, type=int, help="number of device [0,1]")

    # data
    parser.add_argument("--data_root",default= './Datasets', type=str, help="Path to data folder")
    parser.add_argument("--result_root",default = './results_train', type=str, help="Path to save folder")
    parser.add_argument("--experiment", default = 'GitTrain',type=str, help="experiment name")

    # Parameters for network
    parser.add_argument("--norm_use", default=True, type=bool)
    parser.add_argument("--lrelu_use", default=True, type=bool)
    parser.add_argument("--lrelu_slope", default=0.1, type=float)
    parser.add_argument("--zero_padding", default=True, type=bool)
    parser.add_argument("--initial_channel_G", default=48, type=int)
    parser.add_argument("--methods", default='proposed', type=str)

    # Parameters for training
    parser.add_argument("--batch_size", default = 4, type=int)
    parser.add_argument("--output_channel", default = 2, type=int)
    parser.add_argument("--crop_size", default = 256, type=int)
    parser.add_argument("--iterations", default = 100, type=int)
    parser.add_argument("--chk_iter", default = 50, type=int)
    parser.add_argument("--lr_gen", default = 1e-4, type=float)
    parser.add_argument("--lr_decay_epoch", default = 5, type=int)
    parser.add_argument("--lr_decay_rate", default = 0.95, type=float)
    
    # Sampling range for effective pixel size and effective distnace
    parser.add_argument("--dist_min", default=1.353, type=int)
    parser.add_argument("--dist_max", default=5.413, type=int)
    parser.add_argument("--pix_min", default=0.05, type=int)
    parser.add_argument("--pix_max", default=0.25, type=int)

    # Configurations for training dataset 
    parser.add_argument("--wavelength", default=532e-9, type=float)
    parser.add_argument("--pixel_size", default=6.5e-6, type=float)
    parser.add_argument("--numerical_aperture", default = 0.4, type=int)
    parser.add_argument("--magnification", default = 300/9, type=int)  
    
    parser.add_argument("--phase_normalize", default = 2*pi, type=float)

    return parser.parse_args()