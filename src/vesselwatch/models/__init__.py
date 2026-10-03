from .registry import MODEL_NAMES, build_model
from .small_cnn import Net
from .vgg_style import Net_Max

__all__ = ["MODEL_NAMES", "build_model", "Net", "Net_Max"]
