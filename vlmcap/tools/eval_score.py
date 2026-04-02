import os
import json
import numpy as np
import argparse

from PIL import Image
from typing import Dict, Tuple
from metrics import Evaluator
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def coco_caption_eval(coco_gt_root, results_file, output_txt_path):
    coco = COCO(coco_gt_root)
    results_file = os.path.join(results_file, 'result.json')
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.evaluate()

    with open(output_txt_path, 'a', encoding='utf-8') as f:
        f.write("############ Change Caption Metrics ############\n")
        for metric, score in coco_eval.eval.items():
            line = f'{metric}: {score:.4f}'
            print(line)
            f.write(line + '\n')


def jsonl_2_json(pred_file_path):
    json_format_data = []
    jsonl_file = os.path.join(pred_file_path, 'result.jsonl')
    json_file = os.path.join(pred_file_path, 'result.json')

    with open(jsonl_file, 'r') as file_a:
        for line in file_a:
            entry = json.loads(line.strip())
            sample_id = entry.get("image_id")
            pred_response = entry.get("caption")

            b_entry = {
                "image_id": sample_id,
                "caption": pred_response
            }

            json_format_data.append(b_entry)

    with open(json_file, 'w') as output_b_file:
        json.dump(json_format_data, output_b_file, indent=4)


def center_crop_2d(
    arr: np.ndarray,
    crop_size: Tuple[int, int]
) -> np.ndarray:
    """
    对 2D 数组做中心裁剪（不做填充），返回子区域。
    """
    h, w = arr.shape
    ch, cw = crop_size
    if ch > h or cw > w:
        raise ValueError(f"crop_size {crop_size} larger than input size {(h, w)}")
    top  = (h - ch) // 2
    left = (w - cw) // 2
    return arr[top:top+ch, left:left+cw]


def compute_binary_segmentation_metrics(
    label_folder: str,
    pred_folder: str,
    output_txt_path: str = None,
    file_ext: str = '.png',
    rgb_threshold: int = 128,
    crop_size: Tuple[int, int] = (224, 224),
) -> Dict[str, float]:

    evaluator = Evaluator(num_class=2)

    for fname in sorted(os.listdir(label_folder)):
        if not fname.endswith(file_ext):
            continue

        gt_img = Image.open(os.path.join(label_folder, fname)).convert('L')
        gt_np  = np.array(gt_img, dtype=np.uint8)
        gt_np  = center_crop_2d(gt_np, crop_size)
        gt_bin = (gt_np > 0).astype(np.int32)

        pred_img = Image.open(os.path.join(pred_folder, fname)).convert('RGB')
        pred_np  = np.array(pred_img)[..., 0]
        pred_np  = center_crop_2d(pred_np, crop_size)
        pred_bin = (pred_np > rgb_threshold).astype(np.int32)

        evaluator.add_batch(gt_bin, pred_bin)

    pa     = evaluator.Pixel_Accuracy()
    mpa    = evaluator.Pixel_Accuracy_Class()
    mrec   = evaluator.Recall_Class()
    miou, _= evaluator.Mean_Intersection_over_Union()
    fwiou  = evaluator.Frequency_Weighted_Intersection_over_Union()

    results = {
        'Pixel_Accuracy'        : pa,
        'Mean_Pixel_Accuracy'   : mpa,
        'Mean_Recall'           : mrec,
        'mIoU'                  : miou,
        'Frequency_Weighted_IoU': fwiou,
    }

    if output_txt_path:
        with open(output_txt_path, 'w') as f:
            f.write("#### Binary Segmentation Metrics (via Evaluator)\n")
            for k, v in results.items():
                line = f"{k}: {v:.4f}"
                print(line)
                f.write(line + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", type=str, default="/path/to/labels")
    parser.add_argument("--pred_dir", type=str, default="/path/to/preds")
    parser.add_argument("--label_json", type=str, default="/path/to/label_json")
    parser.add_argument("--pred_json_path", type=str, default="/path/to/pred_json")
    parser.add_argument("--output_txt", type=str, default="/path/to/eval_results.txt")
    args = parser.parse_args()

    compute_binary_segmentation_metrics(args.label_dir, args.pred_dir, args.output_txt)

    jsonl_2_json(args.pred_json_path)
    coco_caption_eval(args.label_json, args.pred_json_path, args.output_txt)
    
    print(f"\nAll metrics have been saved to: {args.output_txt}")