import os
import cv2 
import time
import glob
import random 
import argparse
import numpy as np
from piq import MultiScaleSSIMLoss, SSIMLoss, DISTS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


from utils import dict2string, mkdir, get_lr, second2hours
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


def _flow_to_grid(flow):
    bsz, _, h, w = flow.shape
    base_x = torch.arange(w, device=flow.device, dtype=flow.dtype).view(1, 1, w).expand(bsz, h, w)
    base_y = torch.arange(h, device=flow.device, dtype=flow.dtype).view(1, h, 1).expand(bsz, h, w)
    base_coord = torch.stack((base_x / float(w), base_y / float(h)), dim=1)
    warp_map = flow + base_coord
    return (warp_map.permute(0, 2, 3, 1) * 2.0) - 1.0


def _sample_by_grid(image, grid, mode='bilinear'):
    return F.grid_sample(
        image,
        grid.to(image.dtype),
        mode=mode,
        padding_mode='zeros',
        align_corners=False,
    )


def _normalize_flow_mag(mag, mask, quantile=0.95):
    masked_mag = mag * mask
    valid = masked_mag[mask > 0.5]
    if valid.numel() == 0:
        scale = mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    else:
        scale = torch.quantile(valid, quantile).clamp(min=1e-6)
    return (masked_mag / scale).clamp(0, 1)


def _mask_edge_loss(pred_mask, gt_mask):
    pred_dx = pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]
    pred_dy = pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]
    gt_dx = gt_mask[:, :, :, 1:] - gt_mask[:, :, :, :-1]
    gt_dy = gt_mask[:, :, 1:, :] - gt_mask[:, :, :-1, :]

    grad_loss = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

    pred_dx_crop = pred_dx[:, :, :-1, :]
    pred_dy_crop = pred_dy[:, :, :, :-1]
    edge_mask = (pred_dx_crop.abs() + pred_dy_crop.abs()) > 1e-3
    rectilinear_loss = (
        pred_dx_crop.abs() * pred_dy_crop.abs() * edge_mask.float()
    ).sum() / edge_mask.float().sum().clamp(min=1.0)
    return grad_loss + rectilinear_loss


def _compute_dewarping_loss(pred_flow, gt_flow, input_rgb, mask, l1_fn, ms_ssim_fn, weights):
    l1_loss = l1_fn(pred_flow, gt_flow)

    pred_grid = _flow_to_grid(pred_flow)
    with torch.no_grad():
        gt_grid = _flow_to_grid(gt_flow)

    pred_rect = _sample_by_grid(input_rgb, pred_grid, mode='bilinear')
    with torch.no_grad():
        gt_rect = _sample_by_grid(input_rgb, gt_grid, mode='bilinear')
    ms_ssim_loss = ms_ssim_fn(pred_rect, gt_rect)

    pred_mask = _sample_by_grid(mask, pred_grid, mode='nearest')
    with torch.no_grad():
        gt_mask = _sample_by_grid(mask, gt_grid, mode='nearest')
    edge_loss = _mask_edge_loss(pred_mask, gt_mask)

    total_loss = (
        weights['l1'] * l1_loss
        + weights['ms_ssim'] * ms_ssim_loss
        + weights['edge'] * edge_loss
    )
    return total_loss, l1_loss, ms_ssim_loss, edge_loss


def _compute_deshadowing_loss(pred_rgb, gt_rgb, l1_fn, ssim_fn, dists_fn, weights):
    pred_rgb = torch.clamp(pred_rgb, 0, 1)
    gt_rgb = torch.clamp(gt_rgb, 0, 1)

    l1_loss = l1_fn(pred_rgb, gt_rgb)
    ssim_loss = ssim_fn(pred_rgb, gt_rgb)
    dists_loss = dists_fn(pred_rgb, gt_rgb)

    total_loss = (
        weights['l1'] * l1_loss
        + weights['ssim'] * ssim_loss
        + weights['dists'] * dists_loss
    )
    return total_loss, l1_loss, ssim_loss, dists_loss


def _gradient_l1(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def _dice_loss(logits, target, eps=1e-6):
    prob = F.softmax(logits, dim=1)[:, 0]
    target = (target == 0).float()
    inter = (prob * target).sum(dim=(1, 2))
    union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


def _compute_deblurring_loss(pred_rgb, gt_rgb, l1_fn, ms_ssim_fn, weights):
    pred_rgb = torch.clamp(pred_rgb, 0, 1)
    gt_rgb = torch.clamp(gt_rgb, 0, 1)

    l1_loss = l1_fn(pred_rgb, gt_rgb)
    ms_ssim_loss = ms_ssim_fn(pred_rgb, gt_rgb)
    grad_loss = _gradient_l1(pred_rgb, gt_rgb)

    total_loss = (
        weights['l1'] * l1_loss
        + weights['ms_ssim'] * ms_ssim_loss
        + weights['grad'] * grad_loss
    )
    return total_loss, l1_loss, ms_ssim_loss, grad_loss


def _compute_binarization_loss(logits, target, ce_fn, weights):
    ce_loss = ce_fn(logits, target)
    dice_loss = _dice_loss(logits, target)
    total_loss = weights['ce'] * ce_loss + weights['dice'] * dice_loss
    return total_loss, ce_loss, dice_loss


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
        {'task':'dewarping','ratio':1,'im_path':'/home/cl/workspace/dataset/dewarp/doc3d/data/raw/','json_paths':[
            '/home/cl/workspace/dataset/dewarp/doc3d/train.json',
            '/home/cl/workspace/dataset/dewarp/uvdoc/train.json',
        ]},
        {'task':'binarization','ratio':1,'im_path':'/home/cl/workspace/dataset/binarization/','json_paths':['/home/cl/workspace/dataset/binarization/train.json']},
        {'task':'deshadowing','ratio':1,'im_path':'/home/cl/workspace/dataset/deshadowing/','json_paths':['/home/cl/workspace/dataset/deshadowing/train.json']},
        {'task':'appearance','ratio':1,'im_path':'/home/cl/workspace/dataset/appearance/','json_paths':['/home/cl/workspace/dataset/appearance/train.json']}
        ]
    dewarp_datasets_setting = [x for x in all_datasets_setting if x['task'] == 'dewarping']
    trainloaders_dewarp, ratios_dewarp = _build_trainloaders(dewarp_datasets_setting, args)
    trainloaders_all, ratios_all = _build_trainloaders(all_datasets_setting, args)

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
    ms_ssim_loss = MultiScaleSSIMLoss(data_range=1.0, reduction='mean').to(device)
    ssim_loss_fn = SSIMLoss(data_range=1.0, reduction='mean').to(device)
    dists_loss_fn = DISTS(reduction='mean').to(device)
    dewarp_loss_weights = {
        'l1': args.dewarp_l1_weight,
        'ms_ssim': args.dewarp_ms_ssim_weight,
        'edge': args.dewarp_edge_weight,
    }
    deshadow_loss_weights = {
        'l1': args.deshadow_l1_weight,
        'ssim': args.deshadow_ssim_weight,
        'dists': args.deshadow_dists_weight,
    }
    appearance_loss_weights = {
        'l1': args.appearance_l1_weight,
        'ssim': args.appearance_ssim_weight,
        'dists': args.appearance_dists_weight,
    }
    deblur_loss_weights = {
        'l1': args.deblur_l1_weight,
        'ms_ssim': args.deblur_ms_ssim_weight,
        'grad': args.deblur_grad_weight,
    }
    binarization_loss_weights = {
        'ce': args.binarization_ce_weight,
        'dice': args.binarization_dice_weight,
    }

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
        dewarp_l1_loss,dewarp_ms_loss,dewarp_edge_loss = 0,0,0
        deshadow_l1_loss,deshadow_ssim_loss,deshadow_dists_loss = 0,0,0
        app_l1_loss,app_ssim_loss,app_dists_loss = 0,0,0
        deb_l1_loss,deb_ms_loss,deb_grad_loss = 0,0,0
        bin_ce_loss,bin_dice_loss = 0,0
        loss = None
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            task_name = active_trainloaders[loader_index]['task']['task']
            pred_im = model(in_im,task_name)

        if task_name == 'binarization':
            with torch.amp.autocast('cuda', enabled=False):
                gt_mask = gt_im[:, 0, :, :].long()
                binarization_loss, bin_ce_loss, bin_dice_loss = _compute_binarization_loss(
                    pred_im[:, :2, :, :].float(),
                    gt_mask,
                    ce,
                    binarization_loss_weights,
                )
                loss = binarization_loss
        elif task_name == 'appearance':
            with torch.amp.autocast('cuda', enabled=False):
                pred_rgb = torch.clamp(pred_im.float(), 0, 1)
                gt_rgb = torch.clamp(gt_im.float(), 0, 1)
                appearance_loss, app_l1_loss, app_ssim_loss, app_dists_loss = _compute_deshadowing_loss(
                    pred_rgb,
                    gt_rgb,
                    l1,
                    ssim_loss_fn,
                    dists_loss_fn,
                    appearance_loss_weights,
                )
                loss = appearance_loss
        elif task_name == 'deblurring':
            with torch.amp.autocast('cuda', enabled=False):
                pred_rgb = torch.clamp(pred_im.float(), 0, 1)
                gt_rgb = torch.clamp(gt_im.float(), 0, 1)
                deblurring_loss, deb_l1_loss, deb_ms_loss, deb_grad_loss = _compute_deblurring_loss(
                    pred_rgb,
                    gt_rgb,
                    l1,
                    ms_ssim_loss,
                    deblur_loss_weights,
                )
                loss = deblurring_loss

        if task_name == 'dewarping':
            with torch.amp.autocast('cuda', enabled=False):
                pred_flow = pred_im[:, :2, :, :].float()
                gt_flow = gt_im[:, :2, :, :].float()
                input_rgb = torch.clamp(in_im[:, :3, :, :].float(), 0, 1)
                mask = gt_im[:, 2:3, :, :].float()
                dewarping_loss, dewarp_l1_loss, dewarp_ms_loss, dewarp_edge_loss = _compute_dewarping_loss(
                    pred_flow,
                    gt_flow,
                    input_rgb,
                    mask,
                    l1,
                    ms_ssim_loss,
                    dewarp_loss_weights,
                )
                loss = dewarping_loss

        if task_name == 'deshadowing':
            with torch.amp.autocast('cuda', enabled=False):
                pred_rgb = torch.clamp(pred_im.float(), 0, 1)
                gt_rgb = torch.clamp(gt_im.float(), 0, 1)
                deshadowing_loss, deshadow_l1_loss, deshadow_ssim_loss, deshadow_dists_loss = _compute_deshadowing_loss(
                    pred_rgb,
                    gt_rgb,
                    l1,
                    ssim_loss_fn,
                    dists_loss_fn,
                    deshadow_loss_weights,
                )
                loss = deshadowing_loss

        if loss is None:
            raise ValueError(f"Unsupported training task: {task_name}")

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
                pred_flow = pred_im[:, :2, :, :].float()
                gt_flow = gt_im[:, :2, :, :].float()
                flow_mask = gt_im[:, 2:3, :, :].float()
                pred_mag = torch.norm(pred_flow, dim=1, keepdim=True)
                gt_mag = torch.norm(gt_flow, dim=1, keepdim=True)
                pred_vis = _normalize_flow_mag(pred_mag, flow_mask)
                gt_vis = _normalize_flow_mag(gt_mag, flow_mask)
                pred_vis = pred_vis.repeat(1, 3, 1, 1)
                gt_vis = gt_vis.repeat(1, 3, 1, 1)
                writer.add_images(f'Vis/{task_name}/mask', gt_im[:, 2:3, :, :].repeat(1, 3, 1, 1)[:2], total_step)

                pred_rectified = _sample_by_grid(input_vis.float(), _flow_to_grid(pred_flow), mode='bilinear')
                with torch.no_grad():
                    gt_rectified = _sample_by_grid(input_vis.float(), _flow_to_grid(gt_flow), mode='bilinear')
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
        loss_dict['dew_l1_loss']=dewarp_l1_loss.item() if isinstance(dewarp_l1_loss,torch.Tensor) else 0
        loss_dict['dew_ms_loss']=dewarp_ms_loss.item() if isinstance(dewarp_ms_loss,torch.Tensor) else 0
        loss_dict['dew_edge_loss']=dewarp_edge_loss.item() if isinstance(dewarp_edge_loss,torch.Tensor) else 0
        loss_dict['app_loss']=appearance_loss.item() if isinstance(appearance_loss,torch.Tensor) else 0
        loss_dict['app_l1_loss']=app_l1_loss.item() if isinstance(app_l1_loss,torch.Tensor) else 0
        loss_dict['app_ssim_loss']=app_ssim_loss.item() if isinstance(app_ssim_loss,torch.Tensor) else 0
        loss_dict['app_dists_loss']=app_dists_loss.item() if isinstance(app_dists_loss,torch.Tensor) else 0
        loss_dict['des_loss']=deshadowing_loss.item() if isinstance(deshadowing_loss,torch.Tensor) else 0
        loss_dict['des_l1_loss']=deshadow_l1_loss.item() if isinstance(deshadow_l1_loss,torch.Tensor) else 0
        loss_dict['des_ssim_loss']=deshadow_ssim_loss.item() if isinstance(deshadow_ssim_loss,torch.Tensor) else 0
        loss_dict['des_dists_loss']=deshadow_dists_loss.item() if isinstance(deshadow_dists_loss,torch.Tensor) else 0
        loss_dict['deb_loss']=deblurring_loss.item() if isinstance(deblurring_loss,torch.Tensor) else 0
        loss_dict['deb_l1_loss']=deb_l1_loss.item() if isinstance(deb_l1_loss,torch.Tensor) else 0
        loss_dict['deb_ms_loss']=deb_ms_loss.item() if isinstance(deb_ms_loss,torch.Tensor) else 0
        loss_dict['deb_grad_loss']=deb_grad_loss.item() if isinstance(deb_grad_loss,torch.Tensor) else 0
        loss_dict['bin_loss']=binarization_loss.item() if isinstance(binarization_loss,torch.Tensor) else 0
        loss_dict['bin_ce_loss']=bin_ce_loss.item() if isinstance(bin_ce_loss,torch.Tensor) else 0
        loss_dict['bin_dice_loss']=bin_dice_loss.item() if isinstance(bin_dice_loss,torch.Tensor) else 0
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
    parser.add_argument('--dewarp_l1_weight', nargs='?', type=float, default=1.0,
                        help='Weight for dewarping L1 flow loss')
    parser.add_argument('--dewarp_ms_ssim_weight', nargs='?', type=float, default=0.15,
                        help='Weight for dewarping MS-SSIM loss on warped image')
    parser.add_argument('--dewarp_edge_weight', nargs='?', type=float, default=0.08,
                        help='Weight for dewarping mask edge alignment loss')
    parser.add_argument('--deshadow_l1_weight', nargs='?', type=float, default=1.0,
                        help='Weight for deshadowing L1 loss')
    parser.add_argument('--deshadow_ssim_weight', nargs='?', type=float, default=0.25,
                        help='Weight for deshadowing SSIM loss')
    parser.add_argument('--deshadow_dists_weight', nargs='?', type=float, default=0.1,
                        help='Weight for deshadowing DISTS perceptual loss')
    parser.add_argument('--appearance_l1_weight', nargs='?', type=float, default=1.0,
                        help='Weight for appearance L1 loss')
    parser.add_argument('--appearance_ssim_weight', nargs='?', type=float, default=0.25,
                        help='Weight for appearance SSIM loss')
    parser.add_argument('--appearance_dists_weight', nargs='?', type=float, default=0.1,
                        help='Weight for appearance DISTS perceptual loss')
    parser.add_argument('--deblur_l1_weight', nargs='?', type=float, default=1.0,
                        help='Weight for deblurring L1 loss')
    parser.add_argument('--deblur_ms_ssim_weight', nargs='?', type=float, default=0.2,
                        help='Weight for deblurring MS-SSIM loss')
    parser.add_argument('--deblur_grad_weight', nargs='?', type=float, default=0.08,
                        help='Weight for deblurring gradient L1 loss')
    parser.add_argument('--binarization_ce_weight', nargs='?', type=float, default=1.0,
                        help='Weight for binarization cross-entropy loss')
    parser.add_argument('--binarization_dice_weight', nargs='?', type=float, default=0.5,
                        help='Weight for binarization dice loss')
    parser.set_defaults(tboard=False)
    args = parser.parse_args()

    train(args)