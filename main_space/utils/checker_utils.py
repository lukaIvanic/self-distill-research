import os
import torch

import wandb

from main_space.utils.settings_utils import get_wandb_config, get_dataset_config, get_training_config


def strict_error_enable_tip():
    return ("To make misconfigurations like this one a fatal error"
            " instead of a warning, please set 'loggingConfig.strict_error' in settings.py to True.")


def strict_error_disable_tip():
    return ("To allow the script to continue with a warning (if supported),"
            " set 'loggingConfig.strict_error' in settings.py to False. Note that this might affect other "
            "error throwing mechanisms.")


def validatePrecision():

    wandbConfig = get_wandb_config()
    trainingConfig = get_training_config()

    valid_precisions = ["float32", "float16", "bfloat16"]
    if trainingConfig.training_precision not in valid_precisions:
        raise ValueError(
            f"Script Configuration Error: PRECISION must be one of {valid_precisions}, got '{trainingConfig.training_precision}'.")

    if trainingConfig.training_precision == "bfloat16":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported(including_emulation=False):
                print(f"Using natively supported non-emulated bfloat16 precision on CUDA.")
                pass
            elif torch.cuda.is_bf16_supported(including_emulation=True):
                if wandbConfig.loggingConfig.strict_error:
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
            if wandbConfig.loggingConfig.strict_error:
                raise ValueError("Precision bfloat16 was selected, but CUDA was not available, "
                                 "emulation may or may not be supported. "
                                 + strict_error_disable_tip())
            else:
                print("WARNING: Precision bfloat16 was selected, but CUDA was not available, "
                      "emulation may or may not be supported on given device (likely CPU). "
                      + strict_error_enable_tip())

def validate_config():

    wandbConfig = get_wandb_config()
    datasetConfig = get_dataset_config()

    print("Validating configuration...")
    if wandbConfig.is_wandb_enabled and wandb is None:
        raise ImportError(
            "W&B is enabled (ENABLE_WANDB=True) but the 'wandb' library is not installed. Please install it: pip install wandb")

    if wandbConfig.is_profiler_enabled and not wandbConfig.is_wandb_enabled:
        raise ValueError("Configuration Error: ENABLE_PROFILER is True, but ENABLE_WANDB is False. "
                         "The current profiler setup relies on W&B for trace handling. "
                         "If you want to use the profiler, ENABLE_WANDB must also be True.")


    valid_watch_levels = ["all", "gradients", "parameters", "none"]
    if wandbConfig.wandb_watch_level not in valid_watch_levels:
        raise ValueError(
            f"Configuration Error: WANDB_WATCH_LEVEL must be one of {valid_watch_levels}, got '{wandbConfig.wandb_watch_level}'.")

    if not os.path.exists(datasetConfig.tokenizer_path):
        if datasetConfig.tokenizer_path == "path/to/your/tokenizer.json":
            error_msg = (f"ERROR: Tokenizer not found at default placeholder path: '{datasetConfig.tokenizer_path}'. "
                         "Please update TOKENIZER_PATH in the script settings.")
        else:
            error_msg = f"ERROR: Tokenizer not found at: '{datasetConfig.tokenizer_path}'."
        raise FileNotFoundError(error_msg)

    validatePrecision()
    print("Configuration appears valid.")
