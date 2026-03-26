"""
A script for training a diffusion model for paired image-to-image translation.
Uses BraTS20Dataset (our custom loader) instead of the original BRATSVolumes.
"""

import argparse
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          args_to_dict, add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter
from brats_dataset import get_dataloader


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    dist_util.setup_dist(devices=args.devices)

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    datal = get_dataloader(
        data_root=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        num_workers=0,   # keep 0 on Windows to avoid DataLoader multiprocessing issues
        dropout_modality=args.dropout_modality,
    )

    logger.log("Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='i2i',
        contr=args.contr,
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=32,
        out_channels=8,
        image_size=224,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        contr='t1n',
        dropout_modality=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

