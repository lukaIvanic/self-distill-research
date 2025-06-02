import math


def calculate_lr(current_step, peak_lr, warmup_steps, total_training_steps,
                 scheduler_type, min_lr):
    """
    Calculates learning rate with linear warmup and cosine or inverse square root decay.

    Args:
        current_step (int): Current training step (0-indexed).
        peak_lr (float): The maximum learning rate.
        warmup_steps (int): Number of warmup steps.
        total_training_steps (int): Total number of training steps.
        scheduler_type (str): "cosine", "inverse_sqrt", or "linear".
        min_lr (float): For cosine decay and inverse_sqrt, value of learning rate on last iter.

    Returns:
        float: The calculated learning rate for the current step.
    """

    if warmup_steps > total_training_steps:
        raise ValueError(
            f"Warm up steps are higher than total training steps. Warm up: {warmup_steps}, Total training steps: {total_training_steps}")

    if current_step >= total_training_steps:
        raise ValueError(
            f"Current step went over total training steps. Current step: {current_step}, Total training steps: {total_training_steps}")

    if current_step < warmup_steps:
        return peak_lr * ((current_step + 1) / warmup_steps)

    if scheduler_type == "linear":
        return peak_lr

    x = current_step - warmup_steps
    max_x = total_training_steps - warmup_steps

    if scheduler_type == "cosine":
        cos_base = (math.cos((x * math.pi) / max_x) + 1) / 2.0
        lr = cos_base * (peak_lr - min_lr) + min_lr

        return lr

    elif scheduler_type == "inverse_sqrt":
        min_to_peak_ratio_sq = (min_lr / peak_lr) ** 2
        k = (min_to_peak_ratio_sq * max_x) / (1 - min_to_peak_ratio_sq)
        lr = peak_lr * math.sqrt(k / (x + k))
        return max(min_lr, min(lr, peak_lr))
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

def usesAmpOrNot(PRECISION):
    return PRECISION == 'float16' or PRECISION == 'bfloat16'



