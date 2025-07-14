from .kcpm import KCPM

def build_change_detector(config, **kwargs):
    mm_change_hidden_size = config.mm_hidden_size
    
    if "kcpm" in config.mm_change_detector_type:
        return KCPM(in_channels=mm_change_hidden_size)
    else:
        raise ValueError("Not Implement!")