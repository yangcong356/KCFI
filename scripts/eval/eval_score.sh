#!/bin/bash

PRED_BASE_PATH="/your path/KCFI/results/llavanext-LEVIR-openai_clip-vit-large-patch14-Qwen_Qwen2-1.5B-Instruct-mlp2x_gelu-train-5-epoch-32-bs-slicefour_patch-2DWconv-5ksglobal-3kslocal-plus-kcpm-abs_diff-stepDWA-conv-mtl-cd-1e-4-lr-1e-5-plr-1e-5-elr-binary"

python vlmcap/tools/eval_score.py \
    --label_dir /your path/data-rsicc/LEVIR-MCI/images/test/label \
    --pred_dir ${PRED_BASE_PATH}/images \
    --label_json /your path/data-rsicc/LEVIR-MCI/levircc_karpathy_test_gt.json \
    --pred_json_path ${PRED_BASE_PATH} \
    --output_txt ${PRED_BASE_PATH}/eval_score.txt \
