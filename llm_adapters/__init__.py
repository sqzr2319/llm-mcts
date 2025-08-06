from .vllm_adapter import VLLMAdapter
from .debug_model import DebugModel

def get_model_class_by_name(model_name: str):
    """
    Returns the model class based on the model name.
    """
    if "vllm" in model_name.lower():
        return VLLMAdapter
    elif "debug" in model_name.lower():
        return DebugModel
    else:
        raise ValueError(f"Unknown model name: {model_name}")