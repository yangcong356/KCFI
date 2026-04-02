#!/bin/bash
#SBATCH -A XXXXX
#SBATCH -J TIPRe                     # Job name
#SBATCH --nodes=2                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --gres=gpu:4                   # Number of GPUs per node
#SBATCH --cpus-per-task=48              # Number of CPU cores per task
#SBATCH --partition=a100x4             # Partition name
#SBATCH --output=slurm_out.log         # Log file
#SBATCH --time=7-00:00:00                # Max execution time

# Load necessary modules
module purge
module load scl/gcc9.3
module load nvidia/cuda/12.2

# Set environment variables for optimal performance
export OMP_NUM_THREADS=48  # Adjust based on the number of CPU cores

# 核心NCCL设置
export NCCL_IB_DISABLE=0          # 启用InfiniBand
export NCCL_SOCKET_IFNAME=ib0     # 指定IB网卡
export NCCL_DEBUG=INFO

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)  # Use a random port to avoid conflicts
export WORLD_SIZE=$(($SLURM_NNODES * 4))        # Total number of processes
export LOCAL_WORLD_SIZE=4                       # Number of processes per node
export RANK=$SLURM_PROCID                       # Global task rank
export LAUNCHER=pytorch
export TOKENIZERS_PARALLELISM=false

echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "RANK=$RANK"

export LLM_VERSION="Qwen/Qwen2-1.5B-Instruct"
export LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
export VISION_MODEL_VERSION="openai/clip-vit-large-patch14"
export VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
export MMCDTYPE="kcpm"
export PROMPT_VERSION="qwen_2"
export EPOCH=50
export BATCHSIZE=16
export SEGHEAD="conv"
export SELECTED_FEATURE="slicefour_patch"
export EncoderLR=1e-5
export MMProjLR=1e-5
export LR=1e-4
export FUSION_POLICY="abs_diff"

export BASE_RUN_NAME="llavanext-LEVIR-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-train-${EPOCH}-epoch-${BATCHSIZE}-bs-${SELECTED_FEATURE}-${MMCDTYPE}-${FUSION_POLICY}-newDWA-earlUP-MixFFN-${SEGHEAD}-mtl-cd-${LR}-lr-${MMProjLR}-plr-${EncoderLR}-elr-binary"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

srun --nodes=2 --ntasks-per-node=1 bash -c 'torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$LOCAL_WORLD_SIZE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    vlmcap/train/train.py \
    --deepspeed scripts/zero2_fused_adamw.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /your path/data/LEVIR-MCI/ann-llava/LEVIR_CC_cd_train_val.json \
    --image_folder /your path/data/LEVIR-MCI/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_change_detector,mm_seg_head" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_feature ${SELECTED_FEATURE} \
    --mm_change_detector_type ${MMCDTYPE} \
    --mm_img_cd_concat False \
    --mm_fusion_policy ${FUSION_POLICY} \
    --mm_seg_head_type ${SEGHEAD} \
    --mm_num_class 2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /your path/KCFI/checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCHSIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --dataloader_drop_last True \
    --learning_rate ${LR} \
    --mm_projector_lr ${MMProjLR} \
    --mm_vision_tower_lr ${EncoderLR} \
    --weight_decay 0.0005 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    &> /your path/KCFI/training_${BASE_RUN_NAME}.log
    '


# --attn_implementation sdpa
# You can delete the sdpa attn_implementation if you want to use flash attn
