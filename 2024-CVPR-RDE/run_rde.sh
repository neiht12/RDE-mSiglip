root_dir=/kaggle/input/training-rde
tau=0.015 
margin=0.3
noisy_rate=0.0  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=CUHK-PEDES
# CUHK-PEDES ICFG-PEDES RSTPReid

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy

MODEL_NAME="google/siglip-large-patch16-256"

CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name RDE \
    --img_aug \
    --txt_aug \
    --batch_size 32 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}  \
    --num_epoch 45 \
    --pretrain_choice $MODEL_NAME