# Enhancing Perception of Key Changes in Remote Sensing Image Change Captioning

<font size=4><div align='center' > 
[[🤗Paper](https://ieeexplore.ieee.org/abstract/document/11087517)] | [[🤗Code](https://github.com/yangcong356/KCFI.git)] </div></font>


# Introduction
This repository contains the full implementation of the paper **Enhancing Perception of Key Changes in Remote Sensing Image Change Captioning**.

## Quick Workflow

### 1. Environment Setup

```bash
conda create -n kcfi python=3.10 -y
conda activate kcfi
pip install -r requirements.txt
```

Notes:
- Linux + CUDA + multi-GPU environment is recommended.
- The default training scripts use `flash_attention_2`. If your environment does not support it, add `--attn_implementation sdpa` to the training script.

### 2. Prepare Data

Recommended directory structure:

```text
DATA_ROOT/
├── ann-llava/
│   ├── LEVIR_CC_cd_train.json
│   └── LEVIR_CC_cd_test_question.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

Minimal training JSON format:

```json
[
  {
    "id": 0,
    "image": [
      "train/A/sample_0001.png",
      "train/B/sample_0001.png"
    ],
    "label_gray": "train/label/sample_0001.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease describe the changes between the two temporal images."
      },
      {
        "from": "gpt",
        "value": "New buildings appear along both sides of the road."
      }
    ]
  }
]
```

Minimal test JSON format:

```json
[
  {
    "sample_id": 0,
    "image": [
      "test/A/sample_0001.png",
      "test/B/sample_0001.png"
    ],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDescribe the changes between the two images."
      }
    ]
  }
]
```

Notes:
- `image_folder` should point to `DATA_ROOT/images`.
- `label_gray` is only required for training and is used as supervision for change detection.
- `conversations` follows the LLaVA-style dialogue format.

### 3. Update Paths

Replace `/your path` in the following scripts with your local path:
- `scripts/train/local_train.sh`
- `scripts/eval/eval_interleave.sh`
- `scripts/eval/eval_interleave_no_model_base.sh`
- `scripts/eval/eval_score.sh`

If needed, also adjust GPU IDs, batch size, learning rate, or model settings in `scripts/train/local_train.sh`.

### 4. Training

```bash
bash scripts/train/local_train.sh
```

This script calls `vlmcap/train/train.py`, runs joint change captioning and change detection training on `LEVIR-MCI` by default, and saves outputs to `checkpoints/projectors/<run_name>`.

### 5. Inference

If training produces an adapter/projector-only checkpoint, you also need to provide the base model:

```bash
bash scripts/eval/eval_interleave.sh \
  <ckpt_path> \
  <model_base> \
  <data_root> \
  <eval_json_name_without_ext>
```

Example:

```bash
bash scripts/eval/eval_interleave.sh \
  /path/to/checkpoint \
  Qwen/Qwen2-1.5B-Instruct \
  /path/to/LEVIR-MCI \
  LEVIR_CC_cd_test_question
```

If the checkpoint can be loaded directly without a base model, use:

```bash
bash scripts/eval/eval_interleave_no_model_base.sh \
  <ckpt_path> \
  <data_root> \
  <eval_json_name_without_ext>
```

Inference outputs are saved under `results/<name>/`, including:
- `result.jsonl`: generated change captions
- `images/`: predicted binary change maps

### 6. Evaluation

The simplest way is:

```bash
bash scripts/eval/eval_score.sh
```

Or run the evaluation script directly:

```bash
python vlmcap/tools/eval_score.py \
  --label_dir <label_png_dir> \
  --pred_dir <pred_mask_dir> \
  --label_json <coco_gt_json> \
  --pred_json_path <result_dir> \
  --output_txt <output_txt_path>
```

This script reports both:
- change caption metrics: `BLEU`, `METEOR`, `ROUGE_L`, `CIDEr`, etc.

## Key Files

- `vlmcap/train/train.py`: training entrypoint
- `vlmcap/dataset/levircc.py`: training dataset loader
- `vlmcap/eval/model_vqa.py`: inference entrypoint
- `vlmcap/tools/eval_score.py`: joint captioning and segmentation evaluation
- `scripts/train/local_train.sh`: main training script
- `scripts/eval/eval_interleave.sh`: main evaluation script

## Acknowledgements

This project also benefits from the ideas and codebase support of the following repositories:
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Open-CD](https://github.com/likyoo/open-cd)

## Citation

If you find this project useful in your research, please cite:

```bibtex
@ARTICLE{11087517,
  author={Yang, Cong and Li, Zuchao and Jiao, Hongzan and Gao, Zhi and Zhang, Lefei},
  journal={IEEE Transactions on Image Processing},
  title={Enhancing Perception of Key Changes in Remote Sensing Image Change Captioning},
  year={2025},
  volume={34},
  number={},
  pages={7378-7390},
  keywords={Remote sensing;Feature extraction;Accuracy;Large language models;Visualization;Semantics;Tuning;Transformers;Natural language processing;Electronic mail;Multimodal large language model;instruction tuning;remote sensing image change captioning},
  doi={10.1109/TIP.2025.3589096}
}
```
