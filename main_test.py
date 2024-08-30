import os, sys
import random
import matplotlib
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn
from tqdm import tqdm
import functions
import operators

torch.autograd.set_detect_anomaly(True)
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..', 'model'))
matplotlib.use('Agg')

args = functions.InitializationTest.parse_args()
args.batch_size = 1

torch.manual_seed(77)
random.seed(77)

load_path = os.path.join(args.pretrained,args.methods,'model.pth')

cuda = True if torch.cuda.is_available() else False

GPU_NUM = args.device # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check


if __name__ == '__main__':
    
    saving_path = os.path.join(args.result_root,args.experiment)
    
    if args.tissue_type == 'colon':
        
        args.crop_size = 512
        tissue_folder = 'colon_2.4um'
        
    elif args.tissue_type =='appendix':
        
        args.crop_size = 256
        tissue_folder = 'appendix_6.5um'
    
    if args.methods == 'baseline':
        
        Generator = operators.Networks.Field_Generator_Resblk(args,input_channel=1,out_channel=2).to(device=device)
        propagator = operators.ForwardModels.ConventionalForward(args).to(device = device)
        
    elif args.methods == 'proposed':
        
        Generator = operators.Networks.Field_Generator_Resblk(args,input_channel=2,out_channel=2).to(device=device)
        propagator = operators.ForwardModels.EffectiveForward(args).to(device = device)
    
    Generator.load_state_dict(torch.load(load_path, map_location=device)['Field_G_state_dict'])
    Generator.eval()
    
    transform_img_test = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(args.crop_size)])

    test_gt_loader = functions.CustomDataset.DatasetBarg(root=args.data_root, image_set = tissue_folder,
                                                         data_type='field', transform=transform_img_test)

    test_holo_loader =functions.CustomDataset.DatasetBarg(root=args.data_root, image_set = tissue_folder,
                                                          data_type='inten',transform=transform_img_test)
    
    test_holo = iter(DataLoader(test_holo_loader, batch_size=1, shuffle=False))
    test_gt = iter(DataLoader(test_gt_loader, batch_size=1, shuffle=False))
    

    with torch.no_grad():
    
        
        for it in tqdm(range(len(test_holo)),ncols=100):
            
            real_amplitude,real_phase = next(test_gt)
            holo, real_distance,real_pix,real_mag,real_na = next(test_holo)
            
            real_amplitude = real_amplitude.to(device=device).float()
            real_phase = real_phase.to(device=device).float()    

            real_pix = real_pix.to(device=device).float()    
            real_mag = real_mag.to(device=device).float()    
            real_na = real_na.to(device=device).float()    
            real_distance = real_distance.to(device).float()

            effective_pix = real_pix *real_na /real_mag /args.wavelength
            effective_distance = real_distance*1e-3 *real_na**2 /real_mag**2 /args.wavelength

            holo = holo.to(device).float()

            if args.methods == 'baseline':
                
                re_fake, im_fake = Generator(holo)
                fake_amplitude, fake_phase = functions.MyTools.RI2AP(re_fake,im_fake)

            elif args.methods == 'proposed':
                
                field_w_artifact = propagator(holo,effective_distance,effective_pix, return_O = True,back = True)       
                re_field = field_w_artifact.real
                im_field = field_w_artifact.imag

                re_fake, im_fake = Generator(torch.cat([re_field,im_field],dim=1))
                fake_amplitude, fake_phase = functions.MyTools.RI2AP(re_fake,im_fake)
   
            real_phase = real_phase - torch.mean(real_phase)
            fake_phase = fake_phase - torch.mean(fake_phase)
            
            real_phase= real_phase.cpu().detach().numpy()[0][0]
            fake_phase= fake_phase.cpu().detach().numpy()[0][0]

            real_distance = real_distance.squeeze().cpu().detach().numpy()
            real_amplitude = real_amplitude.cpu().detach().numpy()[0][0]                
            fake_amplitude = fake_amplitude.cpu().detach().numpy()[0][0]            
            holo = holo.cpu().detach().numpy()[0][0]

            functions.MyTools.make_path(saving_path)
            functions.MyTools.make_path(os.path.join(saving_path, args.methods))
            functions.MyTools.make_path(os.path.join(saving_path, args.methods, tissue_folder))

            p = os.path.join(saving_path, args.methods, tissue_folder)

            functions.MyTools.savefig(save_path=os.path.join(p, f'test{it+1}.png'),
                                results_data=[holo,real_amplitude, real_phase,fake_amplitude,
                                            fake_phase, real_distance], args=args)
            
            
            
            