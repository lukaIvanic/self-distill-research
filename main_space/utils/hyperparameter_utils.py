import math

from main_space.utils.settings_utils import get_training_config, get_distill_config


def calculate_distill_alpha(config, step_num):

    return get_distill_config().distill_alpha * (step_num / get_training_config().train_steps)

def calculate_lr(step_num, loss=None):
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

    elif trainingConfig.scheduler_type == "custom":
        min_lr = trainingConfig.min_lr
        peak_lr = trainingConfig.peak_lr
        warmup_steps = trainingConfig.warmup_steps
        total_steps = trainingConfig.train_steps


        total_steps_after_warmup = total_steps - warmup_steps
        curr_step_after_warmup = step_num - warmup_steps

        curr_progress = (curr_step_after_warmup / total_steps_after_warmup)

        if step_num < 2000:
            return 1e-3
        elif step_num < 2600:
            return 5e-4
        elif step_num < 3200:
            return 2e-4
        elif step_num < 3600:
            return 1e-4
        elif step_num < 4200:
            return 7e-5
        elif step_num < 5200:
            return 3e-5
        elif step_num < 6000:
            return 1e-5
        elif step_num < 7000:
            return 8e-6
        elif step_num < 8000:
            return 7e-6
        elif step_num < 9000:
            return 5e-6
        elif step_num < 10500:
            return 3e-6
        else:
            return 1e-6

    elif trainingConfig.scheduler_type == "adaptable":

        if loss is None:
            raise BrokenPipeError(f"Loss parameter passed to function hyperparameter_utils.calculate_lr cannot be None when using adaptable learnig rate scheduler.")

        cos_base = (math.cos((x * math.pi) / max_x) + 1) / 2.0
        cosine_lr = cos_base * (trainingConfig.peak_lr - trainingConfig.min_lr) + trainingConfig.min_lr

        target_loss = 2.6
        worst_loss = 3


        loss = min(worst_loss, loss)

        progress = 1 - (worst_loss - loss) / (worst_loss - target_loss)

        max_scale = 2.0
        min_scale = 0.02

        scale = min_scale + (max_scale - min_scale) * progress



        return cosine_lr * scale


    elif trainingConfig.scheduler_type == "1M_specific_2x_train":
        first_stop = 12000
        total_steps = 24000
        leftover = total_steps - first_stop

        x = step_num - trainingConfig.warmup_steps
        max_x = first_stop - trainingConfig.warmup_steps

        if step_num < first_stop:
            cos_base = (math.cos((x * math.pi) / max_x) + 1) / 2.0
            lr = cos_base * (trainingConfig.peak_lr - trainingConfig.min_lr) + trainingConfig.min_lr
            return lr
        else:
            leftover_steps = total_steps - step_num
            org_min_lr = trainingConfig.min_lr
            new_max_lr = org_min_lr * 4.0

            lr = org_min_lr + ((step_num - first_stop)/leftover) * (new_max_lr - org_min_lr)
            return lr



    else:
        raise ValueError(f"Unknown scheduler_type: {trainingConfig.scheduler_type}")

def usesAmpOrNot(PRECISION):
    return PRECISION == 'float16' or PRECISION == 'bfloat16'



