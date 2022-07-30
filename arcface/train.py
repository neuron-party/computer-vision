import argparse
import logging
import os

import torch
import torch.nn as nn
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import torchvision.models as models

# py files
from datasets import *
from losses import *
from lr_scheduler import *
from arcface import *
from utils.logging import *
from utils.callbacks import *
from utils.config import *

try:
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    distributed.init_process_group('nccl')
except:
    world_size = 1
    rank = 0
    distributed.init_proces_group(backend='nccl', init_method='tcp://127.0.0.1:12584', rank=rank, world_size=world_size)
    
def main(args):
    cfg = get_config('configs/custom')
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(args.local_rank)
    
    # init_logging?
    init_logging(rank, cfg.output)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'tensorboard'))
    
    # prepare data/transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, scale=(1.0, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_path = '...'
    test_path = '...'
    train_folder = torchvision.datasets.ImageFolder(train_path, train_transform)
    test_folder = torchvision.datasets.ImageFolder(test_path, test_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_folder,
        num_replicas=world_size,
        rank=args.local_rank,
        shuffle=True,
        seed=cfg.seed
    )
    
    train_loader = DataLoaderX(
        local_rank=args.local_rank,
        dataset=train_folder,
        sampler=train_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(test_folder, shuffle=False, batch_size=cfg.batch_size)
    
    # prepare backbone + model
    backbone = models.resnet101(pretrained=True)
    backbone.fc = nn.Sequential()
    backbone.cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[args.local_rank],
        bucket_cap_mb=16,
        find_unused_parameters=True
    )
    backbone.train()
    backbone._set_static_graph()
    
    # loss + optimizer
    margin_loss = CombinedMarginLoss(
        18,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    arcface_module = ArcfaceModule(
        margin_loss,
        cfg.embedding_size,
        cfg.num_classes,
        cfg.sample_rate,
        cfg.fp16
    )
    arcface_module.train().cuda()
    
    # backbone parameters go before arcface module parameters
    opt = torch.optim.SGD(
        params=[{'params': backbone.parameters()}], [{'params': arcface_module.parameters()}],
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    
    total_batch_size = cfg.batch_size * world_size
    warmup_step = cfg.num_image // total_batch_size * cfg.warmup_epoch
    cfg.num_image = len(train_loader.dataset)
    cfg.total_step = cfg.num_image // total_batch_size * cfg.num_epoch
    
    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=1
    )
    
    # output parameters
    for k, v in cfg.items():
        logging.info(': ' + k + ' ' + str(v))
        
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=0,
        writer=summary_writer
    )
    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    
    # training loop
    start_epoch, global_step = 0, 0
    for epoch in range(start_epoch, cfg.num_epoch):
        train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = arcface_module(local_embeddings, local_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm(backbone.parameters(), 5)
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()
            
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
                
                if global_step % cfg.verbose == 0 and global_step > 0:
                    if rank == 0:
                        backbone.eval()
                        out = testnew(backbone, test_folder, test_loader)
                        print(f'Top 25: {out[:25]}')
                    distributed.barrier()
        
    # todo: need to add save function
    distributed.destroy_process_group()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())