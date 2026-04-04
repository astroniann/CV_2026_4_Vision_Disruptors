"""
Training entrypoint — wires BraTS20Dataset into cwdm's guided_diffusion pipeline.

This replaces the old proof-of-concept main.py (which only tested DWT shapes).
It mirrors cwdm's scripts/train.py exactly, but uses our BraTS20Dataset
instead of cwdm's BRATSVolumes, so your own data pipeline is preserved.

Usage
-----
python main.py \
    --data_dir /path/to/BraTS2024-GLI \
    --contr t1n \
    --devices 0

All other flags have sensible defaults matching cwdm's run.sh.
Run  python main.py --help  to see all options.
"""

import argparse
import random
import sys

import numpy as np
import torch as th

sys.path.insert(0, ".")   # make sure local packages are found

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter

from brats_dataset import get_dataloader


def main():
    args = create_argparser().parse_args()

    # Distributed setup — done once before the loop.
    # setup_dist() has an is_initialized() guard so it is safe here.
    dist_util.setup_dist(devices=args.devices)

    # Train all 4 agents sequentially, each in its own folder
    for contr in ["t1n", "t1c", "t2w", "t2f"]:
        run_dir = f"./runs/agent_{contr}"

        print(f"\n{'='*60}")
        print(f"  Starting agent : {contr.upper()}")
        print(f"  Run directory  : {run_dir}")
        print(f"  Total steps    : {args.lr_anneal_steps}  "
              f"({args.lr_anneal_steps // 1350} epochs x 1350 patients)")
        print(f"  Save interval  : {args.save_interval} steps")
        print(f"{'='*60}\n")

        # Reset random seeds so every agent starts from the same state
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Close the previous logger before creating a new one to avoid
        # leaking open file handles (log.txt, progress.csv) across agents.
        try:
            from guided_diffusion.logger import Logger
            if Logger.CURRENT is not None:
                Logger.CURRENT.close()
        except Exception:
            pass

        # TensorBoard writer — one per agent, writing to run_dir
        summary_writer = None
        if args.use_tensorboard:
            summary_writer = SummaryWriter(log_dir=run_dir)
            summary_writer.add_text(
                "config",
                "\n".join([f"--{k}={repr(v)} <br/>"
                           for k, v in vars(args).items()]),
            )

        # Logger — always configured to run_dir so checkpoints land there
        logger.configure(dir=run_dir)

        # Fresh model + diffusion — no weights shared between agents
        logger.log("Creating model and diffusion...")
        arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
        model, diffusion = create_model_and_diffusion(**arguments)
        model.to(
            dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev()
        )

        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion, maxt=1000
        )

        # Fresh dataloader — reloaded per agent for clean epoch shuffling
        logger.log("Loading dataset...")
        datal = get_dataloader(
            data_root=args.data_dir,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dropout_modality=args.dropout_modality,
        )

        # TrainLoop — lr_anneal_steps stops it after exactly 27000 steps
        logger.log("Starting training...")
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
            mode="i2i",
            contr=contr,
            tumor_loss_weight=args.tumor_loss_weight,
        ).run_loop()

        # Clean up this agent's writer before moving to the next agent
        if summary_writer is not None:
            summary_writer.close()

        logger.log(
            f"Agent {contr.upper()} done. "
            f"Checkpoints saved to {run_dir}/checkpoints/"
        )

    print("\nAll 4 agents finished.")

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        # ---- our additions ----
        seed=42,
        use_tensorboard=True,
        tensorboard_path="",
        dropout_modality=False,   # set True to enable on-the-fly modality dropout
        tumor_loss_weight=10.0,   # weight for tumor voxels in loss; 1.0 = disabled
        # ---- mirrors cwdm run.sh defaults ----
        data_dir="D:/user/BraTS2024-GLI",
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=27000,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=1350,
        resume_checkpoint="",
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="brats",
        devices=[0],
        num_workers=0,
        contr="t1n",        # contrast to synthesise: t1n | t1c | t2w | t2f
        # ---- model/diffusion overrides matching run.sh ----
        num_channels=64,
        channel_mult="1,2,4,4",
        in_channels=32,     # 8 target subbands + 8×3 condition subbands
        out_channels=8,
        image_size=96,
        dims=3,
        num_res_blocks=2,
        num_heads=1,
        num_groups=32,
        attention_resolutions="12,6",
        bottleneck_attention=True,
        resample_2d=False,
        additive_skips=False,
        use_freq=True,
        use_cross_attn=True,
        predict_xstart=True,
        noise_schedule="linear",
        diffusion_steps=1000,
    ))
    defaults["split"] = "train"  # train | validation | additional
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
