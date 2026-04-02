from .conv_seg_head import ConvSegHead


def build_seg_head(config):
  
    if "conv" in config.mm_seg_head_type:
        return ConvSegHead(config.mm_hidden_size, config.mm_num_class, config.proc_crop_size)
    else:
        raise ValueError("Not Implement!")