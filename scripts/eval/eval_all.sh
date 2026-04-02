#!/bin/bash
# 评估
bash scripts/eval/eval_interleave.sh \
    /your path/KCFI/checkpoints/projectors/llavanext-LEVIR-openai_clip-vit-large-patch14-Qwen_Qwen2-1.5B-Instruct-mlp2x_gelu-train-5-epoch-32-bs-slicefour_patch-2DWconv-5ksglobal-3kslocal-plus-kcpm-abs_diff-stepDWA-conv-mtl-cd-1e-4-lr-1e-5-plr-1e-5-elr-binary \
    Qwen/Qwen2-1.5B-Instruct \
    /your path/data-rsicc/LEVIR-MCI \
    LEVIR_CC_cd_test_question