import os
import cv2 
import time
import random 
import argparse
import numpy as np
from tqdm import tqdm
from piq import ssim,psnr
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


from utils import dict2string,mkdir,get_lr,torch2cvimg,second2hours
from loaders import docres_loader
from models import restormer_arch


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
# seed_torch()


def getBasecoord(h,w):
    base_coord0 = np.tile(np.arange(h).reshape(h,1),(1,w)).astype(np.float32)
    base_coord1 = np.tile(np.arange(w).reshape(1,w),(h,1)).astype(np.float32)
    base_coord = np.concatenate((np.expand_dims(base_coord1,-1),np.expand_dims(base_coord0,-1)),-1)
    return base_coord

def train(args):
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(42)

    ### Log file:
    mkdir(args.logdir)
    mkdir(os.path.join(args.logdir,args.experiment_name))
    log_file_path=os.path.join(args.logdir,args.experiment_name,'log.txt')
    log_file=open(log_file_path,'a')
    log_file.write('\n---------------  '+args.experiment_name+'  ---------------\n')
    log_file.close()

    ### Setup tensorboard for visualization
    if args.tboard:
        writer = SummaryWriter(os.path.join(args.logdir,args.experiment_name,'runs'),args.experiment_name)

    ### Setup Dataloader
    all_datasets_setting = [
        {'task':'deblurring','ratio':1,'im_path':'/home/cl/workspace/dataset/deblurring/','json_paths':['/home/cl/workspace/dataset/deblurring/train.json']},
        {'task':'dewarping','ratio':1,'im_path':'/home/cl/workspace/dataset/dewarp/doc3d/data/raw/','json_paths':['/home/cl/workspace/dataset/dewarp/doc3d/train.json']},
        {'task':'binarization','ratio':1,'im_path':'/home/cl/workspace/dataset/binarization/','json_paths':['/home/cl/workspace/dataset/binarization/train.json']},
        {'task':'deshadowing','ratio':1,'im_path':'/home/cl/workspace/dataset/deshadowing/','json_paths':['/home/cl/workspace/dataset/deshadowing/train.json']},
        {'task':'appearance','ratio':1,'im_path':'/home/cl/workspace/dataset/appearance/','json_paths':['/home/cl/workspace/dataset/appearance/train.json']}
        ]
    if args.train_stage == 'dewarp_pretrain':
        datasets_setting = [x for x in all_datasets_setting if x['task'] == 'dewarping']
    else:
        datasets_setting = all_datasets_setting


    ratios = [dataset_setting['ratio'] for dataset_setting in datasets_setting]
    datasets = [docres_loader.DocResTrainDataset(dataset=dataset_setting,img_size=args.im_size) for dataset_setting in datasets_setting]
    trainloaders = []
    for i in range(len(datasets)):
        loader = data.DataLoader(
            dataset=datasets[i],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        trainloaders.append({'task': datasets_setting[i], 'loader': loader, 'iter_loader': iter(loader)})


    ### test loader
    # for i in tqdm(range(args.total_iter)):
    #     loader_index = random.choices(list(range(len(trainloaders))),ratios)[0]
    #     in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])


    ### Setup Model
    model = restormer_arch.Restormer( 
        inp_channels=6, 
        out_channels=3, 
        dim = 48,
        num_blocks = [1,2,2,3],     
        num_refinement_blocks = 1,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.0,
        bias = False,
        LayerNorm_type = 'WithBias',   
        dual_pixel_task = True       
    )
    model = model.to(device)

    ### Optimizer
    optimizer= torch.optim.AdamW(model.parameters(),lr=args.l_rate,weight_decay=5e-4)

    ### LR Scheduler 
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_iter, eta_min=1e-6, last_epoch=-1)

    ### load checkpoint
    iter_start = 0
    if args.resume is not None:
        print("Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')

        model_state = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(model_state, strict=False)

        if args.resume_model_only:
            iter_start = 0
            print("Loaded model weights only; optimizer and scheduler are re-initialized.")
        else:
            if 'optimizer_state' not in checkpoint:
                raise KeyError(
                    "Checkpoint '{}' missing required key 'optimizer_state'".format(args.resume)
                )
            optimizer.load_state_dict(checkpoint['optimizer_state'])

            iter_start = checkpoint.get('iters', checkpoint.get('iter', 0))
            print("Loaded checkpoint '{}' (iter {})".format(args.resume, iter_start))

    ###-----------------------------------------Training-----------------------------------------
    ##initialize
    loss_dict = {}
    total_step = iter_start
    l2 = nn.MSELoss()
    l1 = nn.L1Loss()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    m = nn.Sigmoid()
    best = 0
    best_ce = 999

    ## total_steps
    for iters in range(iter_start,args.total_iter):
        start_time = time.time()
        loader_index = random.choices(list(range(len(trainloaders))),ratios)[0]

        try:
            in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])
        except StopIteration:
            trainloaders[loader_index]['iter_loader']=iter(trainloaders[loader_index]['loader'])
            in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])
        in_im = in_im.float().cuda()
        gt_im = gt_im.float().cuda()

        binarization_loss,appearance_loss,dewarping_loss,deblurring_loss,deshadowing_loss = 0,0,0,0,0
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_im = model(in_im,trainloaders[loader_index]['task']['task'])
            if trainloaders[loader_index]['task']['task'] == 'binarization':
                gt_im = gt_im.long()
                binarization_loss = ce(pred_im[:,:2,:,:], gt_im[:,0,:,:])
                loss = binarization_loss
            elif trainloaders[loader_index]['task']['task'] == 'dewarping':
                dewarping_loss = l1(pred_im[:,:2,:,:], gt_im[:,:2,:,:])
                loss = dewarping_loss
            elif trainloaders[loader_index]['task']['task'] == 'appearance':
                appearance_loss = l1(pred_im, gt_im)
                loss = appearance_loss
            elif trainloaders[loader_index]['task']['task'] == 'deblurring':
                deblurring_loss = l1(pred_im, gt_im)
                loss = deblurring_loss
            elif trainloaders[loader_index]['task']['task'] == 'deshadowing':
                deshadowing_loss = l1(pred_im, gt_im)
                loss = deshadowing_loss

        # TensorBoard image visualization for sampled task.
        if args.tboard and (iters + 1) % args.vis_interval == 0:
            task_name = trainloaders[loader_index]['task']['task']
            input_vis = torch.clamp(in_im[:, :3, :, :], 0, 1)
            prompt_vis = torch.clamp(in_im[:, 3:6, :, :], 0, 1)
            writer.add_images(f'Vis/{task_name}/input', input_vis[:2], total_step)
            writer.add_images(f'Vis/{task_name}/prompt', prompt_vis[:2], total_step)

            if task_name == 'binarization':
                pred_cls = torch.max(torch.softmax(pred_im[:, :2, :, :], 1), 1)[1].unsqueeze(1).float()
                gt_cls = gt_im[:, 0:1, :, :].float()
                pred_vis = pred_cls.repeat(1, 3, 1, 1)
                gt_vis = gt_cls.repeat(1, 3, 1, 1)
            elif task_name == 'dewarping':
                pred_flow = pred_im[:, :2, :, :]
                gt_flow = gt_im[:, :2, :, :]
                pred_mag = torch.norm(pred_flow, dim=1, keepdim=True)
                gt_mag = torch.norm(gt_flow, dim=1, keepdim=True)
                pred_vis = pred_mag / (pred_mag.max() + 1e-6)
                gt_vis = gt_mag / (gt_mag.max() + 1e-6)
                pred_vis = pred_vis.repeat(1, 3, 1, 1)
                gt_vis = gt_vis.repeat(1, 3, 1, 1)
                writer.add_images(f'Vis/{task_name}/mask', gt_im[:, 2:3, :, :].repeat(1, 3, 1, 1)[:2], total_step)

                # Human-readable dewarping preview: apply predicted/gt map to input image.
                bsz, _, h, w = pred_flow.shape
                base_x = torch.arange(w, device=pred_flow.device, dtype=pred_flow.dtype).view(1, 1, w).expand(bsz, h, w)
                base_y = torch.arange(h, device=pred_flow.device, dtype=pred_flow.dtype).view(1, h, 1).expand(bsz, h, w)
                base_coord = torch.stack((base_x / float(w), base_y / float(h)), dim=1)

                pred_map = pred_flow + base_coord
                gt_map = gt_flow + base_coord
                pred_grid = (pred_map.permute(0, 2, 3, 1) * 2.0) - 1.0
                gt_grid = (gt_map.permute(0, 2, 3, 1) * 2.0) - 1.0
                pred_grid = pred_grid.to(input_vis.dtype)
                gt_grid = gt_grid.to(input_vis.dtype)

                pred_rectified = F.grid_sample(input_vis, pred_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                gt_rectified = F.grid_sample(input_vis, gt_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                writer.add_images(f'Vis/{task_name}/pred_rectified', torch.clamp(pred_rectified[:2], 0, 1), total_step)
                writer.add_images(f'Vis/{task_name}/gt_rectified', torch.clamp(gt_rectified[:2], 0, 1), total_step)
            else:
                pred_vis = torch.clamp(pred_im, 0, 1)
                gt_vis = torch.clamp(gt_im, 0, 1)

            writer.add_images(f'Vis/{task_name}/pred', pred_vis[:2], total_step)
            writer.add_images(f'Vis/{task_name}/gt', gt_vis[:2], total_step)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
        loss_dict['dew_loss']=dewarping_loss.item() if isinstance(dewarping_loss,torch.Tensor) else 0
        loss_dict['app_loss']=appearance_loss.item() if isinstance(appearance_loss,torch.Tensor) else 0
        loss_dict['des_loss']=deshadowing_loss.item() if isinstance(deshadowing_loss,torch.Tensor) else 0
        loss_dict['deb_loss']=deblurring_loss.item() if isinstance(deblurring_loss,torch.Tensor) else 0
        loss_dict['bin_loss']=binarization_loss.item() if isinstance(binarization_loss,torch.Tensor) else 0
        end_time = time.time()
        duration = end_time-start_time
        ## log
        if (iters+1) % 10 == 0:
            ## print
            print('iters [{}/{}] -- '.format(iters+1,args.total_iter)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters))))
            ## tbord
            if args.tboard:
                for key,value in loss_dict.items():
                    writer.add_scalar('Train '+key+'/Iterations', value, total_step)
            ## logfile
            with open(log_file_path,'a') as f:
                f.write('iters [{}/{}] -- '.format(iters+1,args.total_iter)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters)))+'\n')


        if (iters+1) % 5000 == 0:
            state = {'iters': iters+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                 os.system('mkdir ' + os.path.join(args.logdir,args.experiment_name))
            torch.save(state, os.path.join(args.logdir,args.experiment_name,"{}.pkl".format(iters+1)))

        sched.step()
        total_step += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--im_size', nargs='?', type=int, default=512, 
                        help='Height of the input image')
    parser.add_argument('--total_iter', nargs='?', type=int, default=100000, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=2e-4, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_model_only', action='store_true',
                        help='Only load model weights when resuming')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--experiment_name', nargs='?', type=str,default='experiment_name',
                        help='the name of this experiment')
    parser.add_argument('--vis_interval', nargs='?', type=int, default=200,
                        help='Iterations interval for tensorboard image visualization')
    parser.add_argument('--train_stage', nargs='?', type=str, default='multitask',
                        choices=['multitask', 'dewarp_pretrain'],
                        help='Training stage selector')
    parser.set_defaults(tboard=False)
    args = parser.parse_args()

    train(args)