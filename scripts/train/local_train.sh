export NUM_GPUS=4
export ADDR=127.0.0.1
export PORT=23001
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

LLM_VERSION="Qwen/Qwen2-1.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
MMCDTYPE=kcpm
PROMPT_VERSION=qwen_2
EPOCH=5
BATCHSIZE=32
SEGHEAD="conv"
SELECTED_FEATURE="slicefour_patch"
EncoderLR=1e-5
MMProjLR=1e-5
LR=1e-4
FUSION_POLICY="abs_diff"

BASE_RUN_NAME="llavanext-LEVIR-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-train-${EPOCH}-epoch-${BATCHSIZE}-bs-${SELECTED_FEATURE}-2DWconv-5ksglobal-3kslocal-plus-${MMCDTYPE}-${FUSION_POLICY}-stepDWA-${SEGHEAD}-mtl-cd-${LR}-lr-${MMProjLR}-plr-${EncoderLR}-elr-binary"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_addr="${ADDR}" --master_port="${PORT}" \
    vlmcap/train/train.py \
    --deepspeed scripts/zero2_offload.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /your path/data-rsicc/LEVIR-MCI/ann-llava/LEVIR_CC_cd_train.json \
    --image_folder /your path/data-rsicc/LEVIR-MCI/images \
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