import math

from main_space.utils.settings_utils import get_training_config, get_distill_config


def calculate_distill_alpha(config, step_num):

    return get_distill_config().distill_alpha * (step_num / get_training_config().train_steps)

def calculate_lr(step_num):
    """
    Calculates learning rate with linear warmup and cosine or inverse square root decay.

    Args:
        step_num (int): Current training step (0-indexed).
        config (main_space.settings.Config): Overall configuration for the training run.
        config.trainingConfig.scheduler_type (str): 'cosine', 'inverse_sqrt' or 'linear'
        config.trainingConfig.learning_rate (float): maximum learning rate, reached after warm-up phase

    Returns:
        float: The calculated learning rate for the current step.
    """

    trainingConfig = get_training_config()

    if trainingConfig.warmup_steps > trainingConfig.train_steps:
        raise ValueError(
            f"Warm up steps are higher than total training steps. Warm up: {trainingConfig.warmup_steps}, Total training steps: {trainingConfig.train_steps}")

    if step_num >= trainingConfig.train_steps:
        raise ValueError(
            f"Current step went over total training steps. Current step: {step_num}, Total training steps: {trainingConfig.train_steps}")

    if step_num < trainingConfig.warmup_steps:
        return trainingConfig.peak_lr * ((step_num + 1) / trainingConfig.warmup_steps)

    if trainingConfig.scheduler_type == "linear":
        return trainingConfig.peak_lr

    x = step_num - trainingConfig.warmup_steps
    max_x = trainingConfig.train_steps - trainingConfig.warmup_steps

    if trainingConfig.scheduler_type == "cosine":
        cos_base = (math.cos((x * math.pi) / max_x) + 1) / 2.0
        lr = cos_base * (trainingConfig.peak_lr - trainingConfig.min_lr) + trainingConfig.min_lr

        return lr

    elif trainingConfig.scheduler_type == "inverse_sqrt":
        min_to_peak_ratio_sq = (trainingConfig.min_lr / trainingConfig.peak_lr) ** 2
        k = (min_to_peak_ratio_sq * max_x) / (1 - min_to_peak_ratio_sq)
        lr = trainingConfig.peak_lr * math.sqrt(k / (x + k))
        return max(trainingConfig.min_lr, min(lr, trainingConfig.peak_lr))
    else:
        raise ValueError(f"Unknown scheduler_type: {trainingConfig.scheduler_type}")

def usesAmpOrNot(PRECISION):
    return PRECISION == 'float16' or PRECISION == 'bfloat16'



