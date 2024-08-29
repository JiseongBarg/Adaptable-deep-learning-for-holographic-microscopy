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

args = functions.InitializationTrain.parse_args()
GPU_NUM = args.device 
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

torch.manual_seed(77)
random.seed(77)

if __name__ == '__main__':
       
    saving_path = os.path.join(args.result_root, args.experiment)

    functions.MyTools.make_path(saving_path)

    txt = open(os.path.join(saving_path,'commandline_args.txt'), 'w')
    stdout = sys.stdout 
    sys.stdout  = txt

    print(args)

    txt.close()

    sys.stdout=stdout
    
    print(args)

    transform_img_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.RandomVerticalFlip(0.5)])

    transform_img_test = transforms.Compose([transforms.ToTensor()])
    
    train_gt_Data = functions.CustomDataset.DatasetBarg(root=args.data_root, data_type='field',
                                            image_set='train',transform=transform_img_train)
    
    valid_gt_Data = functions.CustomDataset.DatasetBarg(root=args.data_root, data_type='field',
                                                image_set='valid', transform=transform_img_test)

    valid_holo_Data = functions.CustomDataset.DatasetBarg(root=args.data_root, data_type='inten', image_set='valid',
                                                transform=transform_img_test)
        
    train_gt_loader = DataLoader(train_gt_Data, batch_size=args.batch_size,shuffle=True)

    N_test = 5 #valid_holo_loader.__len__()    

    scaler = operators.Scaler.ScalerPeff(args).to(device=device)
  
    if args.network == 'Res':
        if args.mode == 'o':
            Generator = operators.Networks.Field_Generator_Resblk(args,input_channel=1,out_channel=2).to(device=device)
        elif args.mode == 't':
            Generator = operators.Networks.Field_Generator_Resblk(args,input_channel=2,out_channel=2).to(device=device)

    propagator = operators.ForwardModels.EffectiveForward(args).to(device = device)

    # optimizer
    op_G = torch.optim.Adam(Generator.parameters(), lr=args.lr_gen, betas=(0.5, 0.9))

    # scheduler
    lr_G = torch.optim.lr_scheduler.StepLR(op_G, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)

    # loss
    criterion_l1 = nn.L1Loss()
    criterion_l1.to(device = device)

    criterion_l2 = nn.MSELoss()
    criterion_l2.to(device = device)
 
    loss_sum_G = 0
    loss_G_list = []
    
    for it in tqdm(range(args.iterations),ncols = 100):
    
        Generator.train()
        
        real_amplitude, real_phase = next(iter(train_gt_loader))
        real_amplitude = real_amplitude.to(device).float()
        real_phase = real_phase.to(device).float()
        
        effective_pix  = torch.rand(size=(args.batch_size, 1, 1, 1)).to(device=device).float()
        effective_pix = (args.pix_max - args.pix_min)*effective_pix + args.pix_min
                
        effective_dist  = torch.rand(size=(args.batch_size, 1, 1, 1)).to(device=device).float()
        effective_dist = (args.dist_max - args.dist_min)*effective_dist + args.dist_min

        real_field, peff = scaler(real_amplitude,real_phase,effective_pix, rand_crop=True,fix_peff = args.pix_fix)
        
        holo = propagator(real_field,effective_dist,peff)

        if args.mode == 'o':
            re_fake, im_fake = Generator(holo)
        elif args.mode == 't':
            field_noise = propagator(holo,effective_dist,peff, return_O = True,back = True)
            re_field = field_noise.real
            im_field = field_noise.imag
            re_fake,im_fake = Generator(torch.cat([re_field,im_field],dim=1))

        re_real = real_field.real
        im_real = real_field.imag

        loss_field = 0.5*(criterion_l1(re_fake,re_real) + criterion_l1(im_fake,im_real)) +0.5*(criterion_l2(re_fake,re_real) + criterion_l2(im_fake,im_real))

        op_G.zero_grad()
        G_loss = 100*loss_field
        G_loss.backward()
        op_G.step()

        loss_sum_G += G_loss.item()

        if (it + 1) % args.chk_iter == 0:

            lr_G.step()
            
            loss_sum_G=round(loss_sum_G/args.chk_iter, 4)
            
            print(f"[{it+1}/{args.iterations}] : L1_loss: {loss_sum_G}")
            
            # path for saving result
            functions.MyTools.make_path(saving_path)
            functions.MyTools.make_path(os.path.join(saving_path, 'generated'))
            
            p = os.path.join(saving_path, 'generated', 'iterations_' + str(it + 1))
            functions.MyTools.make_path(p)

            loss_G_list.append(loss_sum_G)
            
            loss_sum_G = 0

            Generator.eval()
            
            test_gt = iter(DataLoader(valid_gt_Data, batch_size=1, shuffle=False))
            test_holo = iter(DataLoader(valid_holo_Data, batch_size=1, shuffle=False))
            
            with torch.no_grad():
                for b in range(N_test):
                    
                    real_amplitude, real_phase = next(test_gt)
                    holo, real_distance, real_pix, real_mag, real_na = next(test_holo)

                    holo = holo.to(device).float()
                    real_distance = real_distance.to(device).float()
                    real_pix = real_pix.to(device).float()
                    real_mag = real_mag.to(device).float()
                    real_na = real_na.to(device).float()

                    real_amplitude = real_amplitude.to(device=device).float()
                    real_phase = real_phase.to(device=device).float()
                    
                    real_peff = real_pix*real_na / args.wavelength / real_mag
                    real_deff = real_distance*real_na**2 / args.wavelength / real_mag**2

                    if args.mode == 'o':
                        re_fake, im_fake = Generator(holo)
                        fake_field = re_fake + 1j*im_fake
                        fake_amplitude = torch.abs(fake_field)
                        fake_phase = torch.angle(fake_field)

                    elif args.mode == 't':
                        field_noise = propagator(holo,real_deff*1e-3,real_peff, return_O = True,back = True)
                        re_field = field_noise.real
                        im_field = field_noise.imag
                        re_fake, im_fake = Generator(torch.cat([re_field,im_field],dim=1))
                        fake_field = re_fake + 1j*im_fake
                        fake_amplitude = torch.abs(fake_field)
                        fake_phase = torch.angle(fake_field)

                    holo = holo.cpu().detach().numpy()[0][0]
                    real_amplitude = real_amplitude.cpu().detach().numpy()[0][0]
                    real_phase = real_phase.cpu().detach().numpy()[0][0]
                    fake_amplitude = fake_amplitude.cpu().detach().numpy()[0][0]
                    fake_phase = fake_phase.cpu().detach().numpy()[0][0]

                    functions.MyTools.savefig(save_path=os.path.join(p, f'test{b+1}.png'),
                                       results_data=[holo,real_amplitude, real_phase,fake_amplitude,
                                                    fake_phase, real_distance], args=args)

    loss = {}
    loss['G_loss'] = loss_G_list
    save_data = {'iteration': it+1,
                    'Generator_state_dict': Generator.state_dict(),
                    'loss': loss,
                    'args': args}

    torch.save(save_data, os.path.join(saving_path, "model.pth"),_use_new_zipfile_serialization=False)



