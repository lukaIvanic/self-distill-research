import torch
from torch.amp import autocast
from torch.profiler import record_function

from main_space.utils.hyperparameter_utils import usesAmpOrNot, calculate_lr, calculate_distill_alpha
from main_space.utils.settings_utils import get_training_config, get_distill_config
from main_space.utils.log_utils import log_validation_step


# TODO: push this to a new file to share with train_utils.py
def get_loss_classic(model, criterion, batch_input_ids, batch_target_ids):
    with record_function("forward_pass"):
        logits = model(batch_input_ids)

    with record_function("loss_calculation"):
        # The criterion must have reduction='none' to get per-token losses
        per_token_losses = criterion(logits.view(-1, logits.size(-1)), batch_target_ids.view(-1))

        # Calculate the original single average loss
        overall_loss = per_token_losses.mean()

        # Reshape to (batch_size, sequence_length) to analyze loss by position
        losses_by_position = per_token_losses.view(batch_input_ids.size(0), -1)

        # Average across the batch to get a single loss value for each sequence position
        avg_loss_per_position = losses_by_position.mean(dim=0)

        # Reshape into periods of 64 and average each period
        period = 64
        periodic_losses = avg_loss_per_position.view(-1, period).mean(dim=1)

    return overall_loss, periodic_losses

def make_val_step(model, criterion, batch_input_ids, batch_target_ids, device):

    trainingConfig = get_training_config()

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=usesAmpOrNot(trainingConfig.training_precision),
                  dtype=trainingConfig.precision_dtype):

            loss, periodic_losses = get_loss_classic(model, criterion, batch_input_ids, batch_target_ids)

    return loss, periodic_losses

