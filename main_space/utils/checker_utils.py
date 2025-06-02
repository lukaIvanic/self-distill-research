import os
import torch

from torch.amp import GradScaler

import wandb


def strict_error_enable_tip():
    return ("To make misconfigurations like this one a fatal error"
            " instead of a warning, please set 'STRICT_ERROR' in train.py to True.")


def strict_error_disable_tip():
    return ("To allow the script to continue with a warning (if supported),"
            " set 'STRICT_ERROR' in train.py to False. Note that this might affect other "
            "error throwing mechanisms.")


def validatePrecision(PRECISION, STRICT_ERROR):
    valid_precisions = ["float32", "float16", "bfloat16"]
    if PRECISION not in valid_precisions:
        raise ValueError(
            f"Script Configuration Error: PRECISION must be one of {valid_precisions}, got '{PRECISION}'.")

    if PRECISION == "bfloat16":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported(including_emulation=False):
                print(f"Using natively supported non-emulated bfloat16 precision on CUDA.")
                pass
            elif torch.cuda.is_bf16_supported(including_emulation=True):
                if STRICT_ERROR:
                    raise ValueError("Precision bfloat16 was selected, and CUDA was available, "
                                     "but only emulation is supported. "
                                     + strict_error_disable_tip()
                                     )
                else:
                    print("WARNING: Precision bfloat16 was selected, and CUDA was available, "
                          "but only emulation is supported. "
                          "The script will continue, but emulation is very sub-optimal. "
                          + strict_error_enable_tip())
            else:
                raise ValueError("Precision bfloat16 was selected, and CUDA was available, "
                                 "but the device does not support bfloat16 (standard or emulated).")
        else:
            if STRICT_ERROR:
                raise ValueError("Precision bfloat16 was selected, but CUDA was not available, "
                                 "emulation may or may not be supported. "
                                 + strict_error_disable_tip())
            else:
                print("WARNING: Precision bfloat16 was selected, but CUDA was not available, "
                      "emulation may or may not be supported on given device (likely CPU). "
                      + strict_error_enable_tip())


def validate_config(device, ENABLE_WANDB, ENABLE_PROFILER, WANDB_WATCH_LEVEL, TOKENIZER_PATH, PRECISION, STRICT_ERROR):
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

    validatePrecision(PRECISION, STRICT_ERROR=STRICT_ERROR)
    print("Configuration appears valid.")
