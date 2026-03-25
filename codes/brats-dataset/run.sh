#!/bin/bash
# Run script — mirrors cwdm's run.sh exactly.
# Points at our main.py instead of scripts/train.py,
# but all flags and defaults are identical.

# ---- general settings ----
GPU=0
SEED=42
CHANNELS=64
MODE='train'          # train | sample | auto
DATASET='brats'
CONTR='t1n'           # contrast to synthesise: t1n | t1c | t2w | t2f

# ---- sampling/inference settings ----
ITERATIONS=1200
SAMPLING_STEPS=0
RUN_DIR=""

# ---- model settings ----
CHANNEL_MULT=1,2,2,4,4
ADDITIVE_SKIP=False
BATCH_SIZE=1
IMAGE_SIZE=224
IN_CHANNELS=32        # 8 + 8×(number of conditioning modalities)
NOISE_SCHED='linear'

# ---- data paths ----
DATA_DIR="D:\user\BraTS2024-GLI"
if [[ $MODE == 'train' ]]; then
  SPLIT='train'
elif [[ $MODE == 'sample' || $MODE == 'auto' ]]; then
  BATCH_SIZE=1
  SPLIT='validation'
fi

COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=${NOISE_SCHED}
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--additive_skips=${ADDITIVE_SKIP}
--use_freq=False
--predict_xstart=True
--contr=${CONTR}
--split=${SPLIT}
"

TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=100000
--num_workers=0
--devices=${GPU}
--seed=${SEED}
"

if [[ $MODE == 'train' ]]; then
  python main.py $TRAIN $COMMON

elif [[ $MODE == 'sample' ]]; then
  python scripts/sample.py $COMMON \
    --data_dir=${DATA_DIR} \
    --seed=${SEED} \
    --image_size=${IMAGE_SIZE} \
    --use_fp16=False \
    --model_path=${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt \
    --devices=${GPU} \
    --output_dir=./results/${DATASET}_${ITERATIONS}000/ \
    --num_samples=1000 \
    --use_ddim=False \
    --sampling_steps=${SAMPLING_STEPS} \
    --clip_denoised=True

elif [[ $MODE == 'auto' ]]; then
  python scripts/sample_auto.py $COMMON \
    --data_dir=${DATA_DIR} \
    --seed=${SEED} \
    --image_size=${IMAGE_SIZE} \
    --use_fp16=False \
    --model_path=${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt \
    --devices=${GPU} \
    --output_dir=./results/${DATASET}_${ITERATIONS}000/ \
    --num_samples=1000 \
    --use_ddim=False \
    --sampling_steps=${SAMPLING_STEPS} \
    --clip_denoised=True
fi
