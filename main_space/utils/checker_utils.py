import os
import torch

def validate_config(ENABLE_WANDB, wandb, ENABLE_PROFILER, WANDB_WATCH_LEVEL, TOKENIZER_PATH, PRECISION):
    print("Validating configuration...")
    if ENABLE_WANDB and wandb is None:
        raise ImportError(
            "W&B is enabled (ENABLE_WANDB=True) but the 'wandb' library is not installed. Please install it: pip install wandb")

    if ENABLE_PROFILER and not ENABLE_WANDB:
        raise ValueError("Configuration Error: ENABLE_PROFILER is True, but ENABLE_WANDB is False. "
                         "The current profiler setup relies on W&B for trace handling. "
                         "If you want to use the profiler, ENABLE_WANDB must also be True.")

    if ENABLE_PROFILER and wandb is not None and not hasattr(wandb.profiler, 'torch_trace_handler'):
        # This check might be too strict if older wandb versions have different paths
        # but good for ensuring the expected handler exists.
        print("Warning: ENABLE_PROFILER is True, but `wandb.profiler.torch_trace_handler` might not be available. "
              "Ensure your W&B library is up-to-date. Profiling might not work as expected.")

    valid_watch_levels = ["all", "gradients", "parameters", "none"]
    if WANDB_WATCH_LEVEL not in valid_watch_levels:
        raise ValueError(
            f"Configuration Error: WANDB_WATCH_LEVEL must be one of {valid_watch_levels}, got '{WANDB_WATCH_LEVEL}'.")

    if not os.path.exists(TOKENIZER_PATH):
        # Try to give a more helpful message if it's the default path
        if TOKENIZER_PATH == "path/to/your/tokenizer.json":
            error_msg = (f"ERROR: Tokenizer not found at default placeholder path: '{TOKENIZER_PATH}'. "
                         "Please update TOKENIZER_PATH in the script settings.")
        else:
            error_msg = f"ERROR: Tokenizer not found at: '{TOKENIZER_PATH}'."
        raise FileNotFoundError(error_msg)

    valid_precisions = ["float32", "float16", "bfloat16"]
    if PRECISION not in valid_precisions:
        raise ValueError(
            f"Configuration Error: PRECISION must be one of {valid_precisions}, got '{PRECISION}'.")

    if PRECISION != "float32" and not torch.cuda.is_available():
        print(
            f"Warning: PRECISION is set to '{PRECISION}' but CUDA is not available. Pytorch will automaticall use float32 on CPU (?).")

    if PRECISION == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print(
            f"Warning: PRECISION is set to 'bfloat16' but the current CUDA device may not optimally support it or support it at all. Training might be slow or fall back to float32 implicitly by autocast.")

    print("Configuration appears valid.")