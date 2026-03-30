#!/bin/bash

# 使用数组来存放所有的参数，彻底告别反斜杠 \
ARGS=(
  --pretrained_model_name_or_path="Manojb/stable-diffusion-2-1-base"
  --dataset_name="fineface"
  --resolution=512
  --train_batch_size=4
  --dataloader_num_workers=16
  --gradient_accumulation_steps=3
  --mixed_precision="fp16"
  --gradient_checkpointing
  --checkpointing_steps=10000
  --max_train_steps=20000
  --validation_steps=2000
  --learning_rate=1e-04
  --max_grad_norm=1
  --lr_scheduler="constant"
  --lr_warmup_steps=0
  --output_dir="fineface_spatial_id_ours"
  --image_column="image"
  --caption_column="aus"
  --report_to="wandb"
  --rank=32
  --disfa_image_path="/root/autodl-tmp/dataset/disfa/aligned"
  --disfa_label_path="/root/autodl-tmp/dataset/disfa/DISFA/ActionUnit_Labels"
  --disfa_captions_file="/root/autodl-tmp/dataset/disfa/disfa_captions.csv"
  --affectnet_rar_file="/root/autodl-tmp/dataset/affectnet/Manually_Annotated/Manually_Annotated.part01.rar"
  --affectnet_csv_path="/root/autodl-tmp/dataset/affectnet/metadata.csv"
)

# 执行训练脚本并传入上方配置好的参数
python train.py "${ARGS[@]}"
