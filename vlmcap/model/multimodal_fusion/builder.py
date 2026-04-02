from .kcpm import KCPM

def build_change_detector(config):
    mm_change_hidden_size = config.mm_hidden_size
    
    if config.mm_change_detector_type == "kcpm":
        return KCPM(in_channels=mm_change_hidden_size)
    else:
        raise ValueError("Not Implement!")