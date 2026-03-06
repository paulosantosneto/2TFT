# COT DECODING CONFIGS
SEEDS = [42, 43, 44]
DEFAULT_STOP_STRINGS = ["\n\nQ:", "\nQ", "\n\nExercises:", "\n\nQ"]
DEFAULT_K = 10
_MODEL_ALIASES = {
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen2": "Qwen/Qwen2.5-1.5B",
    "phi2": "microsoft/phi-2",
}

# FINETUNING CONFIGS
LORA_TARGET_MODULES_BY_FAMILY = {
    "mistral_llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2":         ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi2":          ['q_proj', 'k_proj', 'v_proj', 'dense', 'fc1', 'fc2'], 
}

def model_family(model_name: str) -> str:
    n = (model_name or "").lower()
    if "phi" in n:
        return "phi2"
    if "qwen2" in n or "qwen-2" in n:
        return "qwen2"
    return "mistral_llama"