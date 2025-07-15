import time
import torch
from torch.amp import autocast
from torch.profiler import record_function

from main_space.utils.hyperparameter_utils import usesAmpOrNot
from main_space.utils.settings_utils import get_dataset_config
from main_space.utils.settings_utils import get_training_config


def get_loss_classic(step_num, device, model, trainingConfig, criterion, batch_input_ids, batch_target_ids):
    with record_function("forward_pass_for_validation"):
        logits = model(batch_input_ids,
                       step_num=step_num,
                       device_type=device.type,
                       amp_enabled=usesAmpOrNot(trainingConfig.training_precision),
                       precision_dtype=trainingConfig.precision_dtype)

    with record_function("validation_loss_calculation"):
        per_token_losses = criterion(logits.view(-1, logits.size(-1)), batch_target_ids.view(-1))
        overall_loss = per_token_losses.mean()
        # losses_by_position = per_token_losses.view(batch_input_ids.size(0), -1)
        # avg_loss_per_position = losses_by_position.mean(dim=0)
        # period = 64
        # periodic_losses = avg_loss_per_position.view(-1, period).mean(dim=1)

    return overall_loss  #, periodic_losses


def make_val_step(model, criterion, batch_input_ids, batch_target_ids, device):
    trainingConfig = get_training_config()

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=usesAmpOrNot(trainingConfig.training_precision),
                      dtype=trainingConfig.precision_dtype):
            val_loss = get_loss_classic(model, device, model, trainingConfig, criterion, batch_input_ids, batch_target_ids)

    return val_loss


def get_next_val_batch(dataset_iter, device, pin_memory_dataloader):
    try:

        batch = next(dataset_iter)

    except StopIteration:
        return None, None


    input_ids = batch['input_ids'].to(device,
                                      non_blocking=True if pin_memory_dataloader and device.type == "cuda" else False)
    target_ids = batch['labels'].to(device,
                                    non_blocking=True if pin_memory_dataloader and device.type == "cuda" else False)

    return input_ids, target_ids


def do_validation_set(model, criterion, dataloader, device):
    """
    :returns average validation loss of whole validation set
    """

    print(f"Starting benchmark on whole validation set...")
    start_time = time.time()

    print(f"\t\tGetting dataset config...")
    datasetConfig = get_dataset_config()
    curr_time = time.time()
    print(f"\t\tGot dataset config in {curr_time-start_time:.2f} seconds.")

    print(f"\t\tGetting dataloader iter...")
    dataset_iter = iter(dataloader)
    print(f"\t\tGot dataloader iter in {time.time() - curr_time:.2f} seconds.")
    curr_time = time.time()


    org_len = len(dataloader)
    print(f"\t\tLen of org dataloader is {org_len}")

    max_a = 5
    total_val_loss = 0

    max_iters = min(len(dataloader) + 1, max_a)

    print(f"\t\tStarting loop..")
    for i in range(max_iters):

        input_ids, target_ids = get_next_val_batch(dataset_iter,
                                                   device,
                                                   datasetConfig.pin_memory_dataloader)

        if input_ids is None or target_ids is None:
            print(f"\t\t\t\tBreaking because None, at iter {i}, len of org dataloader is {org_len}")
            break

        val_loss = make_val_step(model, criterion, input_ids, target_ids,device)

        total_val_loss += val_loss

        print(f"\t\t\t\tVal at iter {i}: {val_loss}, avg_so_far: {total_val_loss/(i+1)}")

    print(f"\t\tLoop lasted for {time.time() - curr_time:.2f} seconds.")

    avg_val_loss = total_val_loss / max_iters
    avg_val_loss = avg_val_loss.item()

    print(f"\t\tRan whole benchmark in {time.time()-start_time:.2f} seconds. Returning avg val loss: {avg_val_loss}")

    return avg_val_loss
