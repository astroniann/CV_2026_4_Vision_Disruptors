"""
A script for automatic multi-contrast MRI synthesis — picks the right model
based on whichever modality is missing in each patient folder.

Uses BraTS20Dataset with dropout_modality=True so batch['missing'] is set
automatically (or point --data_dir at a pseudo-validation set made by
scripts/dropout_modality.py).

Model paths must be filled in below (or passed via --model_t1n etc.)
Weights are available on HuggingFace (see cwdm README).

Usage (Windows):
    python scripts/sample_auto.py ^
        --data_dir C:/data/BraTS2024-GLI ^
        --model_t1n C:/weights/brats_t1n.pt ^
        --model_t1c C:/weights/brats_t1c.pt ^
        --model_t2w C:/weights/brats_t2w.pt ^
        --model_t2f C:/weights/brats_t2f.pt ^
        --output_dir ./results/auto/
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F

sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
from brats_dataset import get_dataloader


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'

    # Map each missing modality to its model weight path
    model_paths = {
        't1n': args.model_t1n,
        't1c': args.model_t1c,
        't2w': args.model_t2w,
        't2f': args.model_t2f,
    }

    # Use our BraTS20Dataset with dropout so batch['missing'] is populated.
    # For a real pseudo-validation set (produced by dropout_modality.py),
    # set dropout_modality=False and point data_dir at the pseudo set folder.
    datal = get_dataloader(
        data_root=args.data_dir,
        split="validation",
        batch_size=args.batch_size,
        num_workers=0,          # keep 0 on Windows
        dropout_modality=args.dropout_modality,
    )

    model.eval()
    idwt = IDWT_3D("haar")
    dwt  = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for batch in iter(datal):
        missing = batch['missing'][0]
        print("Missing modality: {}".format(missing))

        if missing == 'none':
            print("No modality missing in this sample — skipping (use dropout_modality=True or a pseudo-validation set).")
            continue

        selected_model_path = model_paths.get(missing, '')
        if not selected_model_path:
            print(f"No model path set for missing modality '{missing}'. "
                  f"Pass --model_{missing} /path/to/weights.pt")
            continue

        logger.log("Load model from: {}".format(selected_model_path))
        model.load_state_dict(dist_util.load_state_dict(selected_model_path, map_location="cpu"))
        model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())

        batch['t1n'] = batch['t1n'].to(dist_util.dev())
        batch['t1c'] = batch['t1c'].to(dist_util.dev())
        batch['t2w'] = batch['t2w'].to(dist_util.dev())
        batch['t2f'] = batch['t2f'].to(dist_util.dev())

        # batch['subj'] is the full path — use just the patient folder name
        subj = pathlib.Path(batch['subj'][0]).name
        print(subj)

        if missing == 't1n':
            cond_1 = batch['t1c']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
        elif missing == 't1c':
            cond_1 = batch['t1n']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
        elif missing == 't2w':
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2f']
        elif missing == 't2f':
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2w']
        else:
            print("This contrast can't be synthesized.")
            continue

        # Conditioning vector
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
        cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_2)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_3)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        noise = th.randn(args.batch_size, 8, 112, 112, 80).to(dist_util.dev())

        sample = diffusion.p_sample_loop(
            model=model,
            shape=noise.shape,
            noise=noise,
            cond=cond,
            clip_denoised=args.clip_denoised,
            model_kwargs={},
        )

        B, _, D, H, W = sample.size()
        sample = idwt(
            sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
            sample[:, 1, :, :, :].view(B, 1, D, H, W),
            sample[:, 2, :, :, :].view(B, 1, D, H, W),
            sample[:, 3, :, :, :].view(B, 1, D, H, W),
            sample[:, 4, :, :, :].view(B, 1, D, H, W),
            sample[:, 5, :, :, :].view(B, 1, D, H, W),
            sample[:, 6, :, :, :].view(B, 1, D, H, W),
            sample[:, 7, :, :, :].view(B, 1, D, H, W),
        )

        sample[sample <= 0.04] = 0

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)

        # Pad back to 240×240 and crop to 155 slices
        pad_sample = F.pad(sample, (0, 0, 8, 8, 8, 8), mode='constant', value=0)
        sample = pad_sample[:, :, :, :155]

        # Save synthesised volume in-place alongside the existing modalities
        out_dir = pathlib.Path(args.output_dir) / subj
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(sample.shape[0]):
            # Name matches BraTS2024 convention so downstream tools recognise it
            out_name = str(out_dir / f"{subj}-{missing}.nii.gz")
            nib.save(
                nib.Nifti1Image(sample.detach().cpu().numpy()[i], np.eye(4)),
                out_name,
            )
            print(f'Saved to {out_name}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        # Per-modality model weight paths — fill in or pass on command line
        model_t1n="",
        model_t1c="",
        model_t2w="",
        model_t2f="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=224,
        half_res_crop=False,
        concat_coords=False,
        contr="",
        dropout_modality=True,   # True = use on-the-fly dropout; False = use pre-made pseudo-val set
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


















