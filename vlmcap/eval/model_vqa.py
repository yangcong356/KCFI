import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import json
from tqdm import tqdm
import numpy as np

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils import disable_torch_init
from mm_utils import get_model_name_from_path

from typing import Dict
import transformers
import re

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def save_tensor_as_rgb_png(tensor, path, filename):
    assert tensor.shape == (1, 224, 224) and tensor.dtype == torch.int64, "The input Tensor must be of type int64 in the shape of [1, 224, 224]."

    tensor = tensor.squeeze(0)

    color_map = {
        0: (0, 0, 0),   # black
        1: (255, 255, 255)  # white
    }

    rgb_array = np.zeros((224, 224, 3), dtype=np.uint8)

    for key, color in color_map.items():
        mask = (tensor == key).cpu().numpy()
        rgb_array[mask] = color

    image = Image.fromarray(rgb_array)

    image.save(os.path.join(path, filename))

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    system_tokens = tokenizer("system").input_ids + nl_tokens

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id = [im_start] + system_tokens + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    for sentence in source:
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split("<image>")
            encoded = tokenizer(role).input_ids + nl_tokens
            for idx, fragment in enumerate(texts):
                encoded += tokenizer(fragment).input_ids
                if idx < len(texts) - 1:
                    encoded += [IMAGE_TOKEN_INDEX] + nl_tokens
            encoded += [im_end] + nl_tokens
            assert sum(token == IMAGE_TOKEN_INDEX for token in encoded) == num_image
        elif sentence["value"] is None:
            encoded = tokenizer(role).input_ids + nl_tokens
        else:
            encoded = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += encoded

    return torch.tensor([input_id], dtype=torch.long)
def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    img_save_path = os.path.join(os.path.dirname(answers_file), "images_"+str(args.chunk_idx))
    os.makedirs(img_save_path, exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["sample_id"]

        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        img_save_file_name = os.path.basename(image_files[0])

        args.conv_mode = "qwen_2"

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        input_ids = preprocess_qwen([line["conversations"][0],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()

        image_tensors = []
        for image_file in image_files:
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
        # image_tensors = torch.cat(image_tensors, dim=0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        with torch.inference_mode():
            output_ids, change_logits = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        change_pred = torch.argmax(change_logits, dim=1)
        save_tensor_as_rgb_png(change_pred, img_save_path, img_save_file_name)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({
                                   "image_id": idx,
                                   "caption": outputs,
                                   }) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)