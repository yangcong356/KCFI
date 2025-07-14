#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_fusion.builder import build_change_detector
from .multimodal_projector.builder import build_vision_projector
from .seg_head.builder import build_seg_head

from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from mm_utils import get_anyres_image_grid_shape
from utils import rank0_print, rank_print
import random

import pdb


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.change_detector = build_change_detector(config)
            self.seg_head = build_seg_head(config)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type


        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.use_mm_cd = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.mm_num_patches_per_side = vision_tower.num_patches_per_side
        self.config.mm_change_detector_type = getattr(model_args, "mm_change_detector_type", "fdfa")
        self.config.mm_fusion_policy = model_args.mm_fusion_policy
        self.config.mm_img_cd_concat = model_args.mm_img_cd_concat
        self.config.mm_seg_head_type = model_args.mm_seg_head_type
        self.config.mm_num_class = model_args.mm_num_class

        if getattr(self, "change_detector", None) is None:
            self.change_detector = build_change_detector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.change_detector.parameters():
                p.requires_grad = True

        if getattr(self, "seg_head", None) is None:
            self.seg_head = build_seg_head(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.seg_head.parameters():
                p.requires_grad = True

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            proj_incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {proj_incompatible_keys}")


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()


    def generate_indices(self,N):
        assert N % 2 == 0, "N must be even for this reordering scheme"
        first_half = torch.arange(0, N, 2)
        second_half = torch.arange(1, N, 2)
        indices = torch.cat([first_half, second_half])
        return indices


    def encode_images(self, images):

        if self.config.mm_fusion_policy is not None:
            image_features = self.get_model().get_vision_tower()(images)
            B, N, _ = image_features.shape
            # 动态生成和应用索引，使特征顺序变成[A,A,...,B,B]
            indices = self.generate_indices(B)
            original_indices = torch.argsort(indices)
            image_features = image_features[indices]

            # 分割图像特征进行变化检测
            pre, post = image_features[:B//2], image_features[B//2:]
            # num_blocks = N // self.config.mm_num_patches_per_side

            # # 生成mask，交替选择需要交换的块
            # mask = torch.empty(N, dtype=torch.bool)
            # mask[::num_blocks] = True

            # # 使用mask进行交换
            # pre[:, mask, :], post[:, mask, :] = post[:, mask, :], pre[:, mask, :]

            change_features = self.get_model().change_detector(pre, post, self.config.mm_fusion_policy)
            change_logits = self.get_model().seg_head(change_features)

            if self.config.mm_img_cd_concat:
                
                # 恢复原始的交叉顺序[A,B,...,A,B]
                image_features = image_features[original_indices]
                image_features = self.get_model().mm_projector(image_features)
                change_features_proj = self.get_model().mm_projector(change_features)
                _, _, C = image_features.shape
                
                # 将图像特征与变化特征拼接成[A,B,C,...,A,B,C]
                image_features = image_features.view(B//2, 2, N, C)
                change_features_proj = change_features_proj.view(B//2, 1, N, C)
                image_features = torch.cat([image_features, change_features_proj], dim=1).view(-1, N, C)

                return image_features, change_logits
            else:
                change_features_proj = self.get_model().mm_projector(change_features)
                return change_features_proj, change_logits
        else:
            image_features = self.get_model().mm_projector(image_features)
            return image_features


        # image_features = self.get_model().get_vision_tower()(images)
        # B, _, _ = image_features.shape

        # if self.config.mm_fusion_policy is not None:
            
        #     # dynamic generate indices
        #     indices = self.generate_indices(image_features.shape[0])

        #     # restore the original indices
        #     original_indices = torch.argsort(indices)

        #     image_features=image_features[indices]
        #     pre = image_features[:int(B/2),:]
        #     post = image_features[int(B/2):,:]
        #     change_features = self.get_model().change_detector(pre, post, self.config.mm_fusion_policy)
            
        #     change_logits = self.get_model().seg_head(change_features)

        #     if self.config.mm_img_cd_concat == True:

        #         image_features = image_features[original_indices]
        #         image_features = self.get_model().mm_projector(image_features)
        #         change_features_proj = self.get_model().mm_projector(change_features)
        #         _, N, C = image_features.shape

        #         # (A,B,...,A,B) to (A,B,C,...,A,B,C)
        #         image_features = image_features.view(int(B/2), 2, N, C)
        #         temp_image_features = []
        #         for i in range(image_features.shape[0]):
        #             img1, img2 = image_features[i]
        #             cf = change_features_proj[i]
        #             temp_image_features.extend([img1, img2, cf])
        #         image_features = torch.stack(temp_image_features).view(-1, N, C)

        #         return image_features, change_logits
            # if self.config.mm_img_cd_concat == True:

            #     image_features = image_features[original_indices]
            #     _, N, C = image_features.shape
            #     # (A,B,...,A,B) to (A,B,C,...,A,B,C)
            #     image_features = image_features.view(int(B/2), 2, N, C)
            #     temp_image_features = []
            #     for i in range(image_features.shape[0]):
            #         img1, img2 = image_features[i]
            #         cf = change_features[i]
            #         temp_image_features.extend([img1, img2, cf])
            #     image_features = torch.stack(temp_image_features).view(-1, N, C)

            #     image_features = self.get_model().mm_projector(image_features)
            #     return image_features

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images] # 2*B, 1, C, H, W

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            if self.config.mm_fusion_policy is not None:
                split_sizes = split_sizes[:int(len(split_sizes)/2)]
                if self.config.mm_img_cd_concat == True:
                    split_sizes = split_sizes * 3
            
            if self.config.mm_fusion_policy is not None:
                encoded_image_features, change_logits = self.encode_images(concat_images)
            else:
                encoded_image_features = self.encode_images(concat_images)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                image_features.append(image_feat)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        # pdb.set_trace()
        if self.config.mm_fusion_policy is not None:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, change_logits
        else:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False