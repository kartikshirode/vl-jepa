"""VL-JEPA: Vision-Language Joint Embedding Predictive Architecture for Jetson Orin Nano"""

__version__ = "0.1.0"

from .models.vl_jepa import VLJEPAModel, create_vl_jepa_model
from .models.vision_encoder import VisionEncoder, VisionEncoderWithProjection
from .models.text_encoder import TextEncoder, TextEncoderWithPooling
from .models.predictor import PredictorMLP, PredictorTransformer, PredictorWithCrossAttention

__all__ = [
    "VLJEPAModel",
    "create_vl_jepa_model",
    "VisionEncoder",
    "VisionEncoderWithProjection",
    "TextEncoder",
    "TextEncoderWithPooling",
    "PredictorMLP",
    "PredictorTransformer",
    "PredictorWithCrossAttention",
]
