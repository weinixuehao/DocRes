import os
import cv2 
import time
import glob
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
from inference import dewarp_prompt, deshadow_prompt, appearance_prompt, deblur_prompt
from data.preprocess.crop_merge_image import stride_integral


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


def _build_trainloaders(datasets_setting, args):
    ratios = [dataset_setting['ratio'] for dataset_setting in datasets_setting]
    datasets = [docres_loader.DocResTrainDataset(dataset=dataset_setting, img_size=args.im_size) for dataset_setting in datasets_setting]
    trainloaders = []
    for i in range(len(datasets)):
        loader = data.DataLoader(
            dataset=datasets[i],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError(
                "Empty DataLoader for task '{}': dataset too small for batch_size={} with drop_last=True".format(
                    datasets_setting[i]['task'], args.batch_size
                )
            )
        trainloaders.append({'task': datasets_setting[i], 'loader': loader, 'iter_loader': iter(loader)})
    return trainloaders, ratios


def _train_stage_for_iter(iter_idx, stage1_iter):
    return 'dewarp_pretrain' if iter_idx < stage1_iter else 'multitask'


def _is_image_file(path):
    lower_path = path.lower()
    return lower_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))


def _collect_eval_images():
    eval_input_dir = './input/test'
    if not os.path.isdir(eval_input_dir):
        return []
    all_paths = sorted(glob.glob(os.path.join(eval_input_dir, '*')))
    image_paths = [p for p in all_paths if _is_image_file(p)]
    return image_paths


def _to_model_input(image_with_prompt, device):
    image_with_prompt = image_with_prompt.astype(np.float32) / 255.0
    return torch.from_numpy(image_with_prompt.transpose(2, 0, 1)).unsqueeze(0).to(
        device=device,
        dtype=torch.float32
    )


def _predict_uint8_image(model, model_input, task_name):
    with torch.no_grad():
        pred = model(model_input, task_name)
        pred = torch.clamp(pred, 0, 1)
        pred = pred[0].permute(1, 2, 0).cpu().numpy()
    return (pred * 255).astype(np.uint8)


def _infer_dewarp_stage(model, device, image_bgr, return_mask=False):
    input_size = 384
    im_org = image_bgr
    im_masked, prompt_org = dewarp_prompt(im_org.copy())
    h, w = im_masked.shape[:2]

    im_masked = cv2.resize(im_masked, (input_size, input_size)).astype(np.float32) / 255.0
    im_masked = torch.from_numpy(im_masked.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)

    prompt = torch.from_numpy(prompt_org.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    in_im = torch.cat((im_masked, prompt), dim=1)

    base_coord = getBasecoord(input_size, input_size) / input_size
    with torch.no_grad():
        pred = model(in_im, 'dewarping')
        pred = pred[0][:2].permute(1, 2, 0).cpu().numpy()
        pred = pred + base_coord

    for _ in range(15):
        pred = cv2.blur(pred, (3, 3), borderType=cv2.BORDER_REPLICATE)
    pred = cv2.resize(pred, (w, h)) * (w, h)
    pred = pred.astype(np.float32)
    out_im = cv2.remap(im_org, pred[:, :, 0], pred[:, :, 1], cv2.INTER_LINEAR)
    if not return_mask:
        return out_im

    # prompt_org third channel stores dewarp foreground mask in [0, 1].
    mask = np.clip(prompt_org[:, :, 2] * 255.0, 0, 255).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return out_im, mask


def _infer_shading_like_stage(model, device, image_bgr, task_name):
    max_size = 1600
    im_org = image_bgr
    h, w = im_org.shape[:2]
    prompt = deshadow_prompt(im_org) if task_name == 'deshadowing' else appearance_prompt(im_org)
    in_im = np.concatenate((im_org, prompt), axis=-1)

    use_stride = max(w, h) < max_size
    if use_stride:
        in_im, padding_h, padding_w = stride_integral(in_im, 8)
    else:
        in_im = cv2.resize(in_im, (max_size, max_size))

    in_im = _to_model_input(in_im, device)
    pred = _predict_uint8_image(model, in_im, task_name)
    if use_stride:
        out_im = pred[padding_h:, padding_w:]
    else:
        pred[pred == 0] = 1
        shadow_map = cv2.resize(im_org, (max_size, max_size)).astype(np.float32) / pred.astype(np.float32)
        shadow_map = cv2.resize(shadow_map, (w, h))
        shadow_map[shadow_map == 0] = 0.00001
        out_im = np.clip(im_org.astype(np.float32) / shadow_map, 0, 255).astype(np.uint8)
    return out_im


def _infer_deblur_stage(model, device, image_bgr):
    im_org = image_bgr
    in_im, padding_h, padding_w = stride_integral(im_org, 8)
    prompt = deblur_prompt(in_im)
    in_im = np.concatenate((in_im, prompt), axis=-1)
    in_im = _to_model_input(in_im, device)
    pred = _predict_uint8_image(model, in_im, 'deblurring')
    out_im = pred[padding_h:, padding_w:]
    return out_im


def run_ckpt_visual_inference(model, device, args, current_iter, log_file_path, current_train_stage):
    eval_images = _collect_eval_images()
    if not eval_images:
        message = '[ckpt-eval] iter {}: skip, no valid images in ./input/test\n'.format(current_iter)
        print(message.strip())
        with open(log_file_path, 'a') as f:
            f.write(message)
        return

    save_root = os.path.join(
        args.logdir,
        args.experiment_name,
        'test',
        f'iter_{current_iter:06d}'
    )
    mkdir(save_root)

    was_training = model.training
    model.eval()
    model.float()

    for img_path in eval_images:
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError('cv2.imread failed')

            image_name = os.path.splitext(os.path.basename(img_path))[0]
            sample_dir = os.path.join(save_root, image_name)
            mkdir(sample_dir)
            cv2.imwrite(os.path.join(sample_dir, 'input.png'), image)

            stage, dewarp_mask = _infer_dewarp_stage(model, device, image, return_mask=True)
            cv2.imwrite(os.path.join(sample_dir, '00_mask.png'), dewarp_mask)
            cv2.imwrite(os.path.join(sample_dir, '01_dewarp.png'), stage)

            if current_train_stage == 'dewarp_pretrain':
                stage_pipeline = []
            else:
                stage_pipeline = [
                    ('02_deshadow.png', lambda im: _infer_shading_like_stage(model, device, im, 'deshadowing')),
                    ('03_appearance.png', lambda im: _infer_shading_like_stage(model, device, im, 'appearance')),
                    ('04_deblur.png', lambda im: _infer_deblur_stage(model, device, im)),
                ]
            for output_name, stage_fn in stage_pipeline:
                stage = stage_fn(stage)
                cv2.imwrite(os.path.join(sample_dir, output_name), stage)
        except Exception as exc:
            err_message = f'[ckpt-eval] iter {current_iter}: failed on {img_path} | {exc}\n'
            print(err_message.strip())
            with open(log_file_path, 'a') as f:
                f.write(err_message)

    done_message = f'[ckpt-eval] iter {current_iter}: saved to {save_root}\n'
    print(done_message.strip())
    with open(log_file_path, 'a') as f:
        f.write(done_message)

    if was_training:
        model.train()


def train(args):
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(42)
    if args.stage1_iter >= args.total_iter:
        raise ValueError("stage1_iter must be smaller than total_iter")

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
    dewarp_datasets_setting = [x for x in all_datasets_setting if x['task'] == 'dewarping']
    trainloaders_dewarp, ratios_dewarp = _build_trainloaders(dewarp_datasets_setting, args)
    trainloaders_all, ratios_all = _build_trainloaders(all_datasets_setting, args)


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
        iter_start = checkpoint.get('iters', checkpoint.get('iter', 0))
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            sched.load_state_dict(checkpoint['scheduler_state'])
        print("Loaded checkpoint '{}' (iter {})".format(args.resume, iter_start))

    ###-----------------------------------------Training-----------------------------------------
    ##initialize
    loss_dict = {}
    total_step = iter_start
    l1 = nn.L1Loss()
    ce = nn.CrossEntropyLoss()

    ## total_steps
    last_stage_name = None
    for iters in range(iter_start,args.total_iter):
        start_time = time.time()
        stage_name = _train_stage_for_iter(iters, args.stage1_iter)
        if stage_name != last_stage_name:
            phase_msg = f"Switch training stage at iter {iters}: {stage_name}"
            print(phase_msg)
            with open(log_file_path, 'a') as f:
                f.write(phase_msg + '\n')
            last_stage_name = stage_name

        active_trainloaders = trainloaders_dewarp if stage_name == 'dewarp_pretrain' else trainloaders_all
        active_ratios = ratios_dewarp if stage_name == 'dewarp_pretrain' else ratios_all

        loader_index = random.choices(list(range(len(active_trainloaders))),active_ratios)[0]

        try:
            in_im,gt_im = next(active_trainloaders[loader_index]['iter_loader'])
        except StopIteration:
            active_trainloaders[loader_index]['iter_loader']=iter(active_trainloaders[loader_index]['loader'])
            in_im,gt_im = next(active_trainloaders[loader_index]['iter_loader'])
        in_im = in_im.float().cuda()
        gt_im = gt_im.float().cuda()

        binarization_loss,appearance_loss,dewarping_loss,deblurring_loss,deshadowing_loss = 0,0,0,0,0
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            task_name = active_trainloaders[loader_index]['task']['task']
            pred_im = model(in_im,task_name)
            if task_name == 'binarization':
                gt_im = gt_im.long()
                binarization_loss = ce(pred_im[:,:2,:,:], gt_im[:,0,:,:])
                loss = binarization_loss
            elif task_name == 'dewarping':
                dewarping_loss = l1(pred_im[:,:2,:,:], gt_im[:,:2,:,:])
                loss = dewarping_loss
            elif task_name == 'appearance':
                appearance_loss = l1(pred_im, gt_im)
                loss = appearance_loss
            elif task_name == 'deblurring':
                deblurring_loss = l1(pred_im, gt_im)
                loss = deblurring_loss
            elif task_name == 'deshadowing':
                deshadowing_loss = l1(pred_im, gt_im)
                loss = deshadowing_loss

        # TensorBoard image visualization for sampled task.
        if args.tboard and (iters + 1) % args.vis_interval == 0:
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
        sched.step()
    
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
            print('iters [{}/{}] [{}] -- '.format(iters+1,args.total_iter,stage_name)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters))))
            ## tbord
            if args.tboard:
                for key,value in loss_dict.items():
                    writer.add_scalar('Train '+key+'/Iterations', value, total_step)
            ## logfile
            with open(log_file_path,'a') as f:
                f.write('iters [{}/{}] [{}] -- '.format(iters+1,args.total_iter,stage_name)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters)))+'\n')


        if (iters+1) % 5000 == 0:
            state = {'iters': iters+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),
                     'scheduler_state': sched.state_dict(),}
            if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                 os.system('mkdir ' + os.path.join(args.logdir,args.experiment_name))
            ckpt_path = os.path.join(args.logdir,args.experiment_name,"{}.pkl".format(iters+1))
            torch.save(state, ckpt_path)
            run_ckpt_visual_inference(
                model=model,
                device=device,
                args=args,
                current_iter=iters+1,
                log_file_path=log_file_path,
                current_train_stage=stage_name
            )
        total_step += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--im_size', nargs='?', type=int, default=384, 
                        help='Height of the input image')
    parser.add_argument('--total_iter', nargs='?', type=int, default=350000, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10, 
                        help='Batch Size')
    parser.add_argument('--num_workers', nargs='?', type=int, default=2,
                        help='Dataloader workers')
    parser.add_argument('--l_rate', nargs='?', type=float, default=2e-4, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--experiment_name', nargs='?', type=str,default='experiment_name',
                        help='the name of this experiment')
    parser.add_argument('--vis_interval', nargs='?', type=int, default=200,
                        help='Iterations interval for tensorboard image visualization')
    parser.add_argument('--stage1_iter', nargs='?', type=int, default=100000,
                        help='Train dewarping only before this iteration')
    parser.set_defaults(tboard=False)
    args = parser.parse_args()

    train(args)