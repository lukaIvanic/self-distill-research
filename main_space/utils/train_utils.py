
import torch
from torch.amp import autocast
from torch.profiler import record_function

import wandb

from main_space.utils.log_utils import step_log
from main_space.utils.hyperparameter_utils import usesAmpOrNot, calculate_lr



def set_step_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_train_step(config, step_num, model, criterion, optimizer, device,
                    batch_input_ids, batch_target_ids,  # Directly supplied
                    wandb_run_obj, profiler_obj
                    ):

    trainingConfig = config.trainingConfig


    current_actual_lr = calculate_lr(
        current_step=step_num,
        peak_lr=trainingConfig.learning_rate,  # LEARNING_RATE from config is the peak LR
        warmup_steps=trainingConfig.warmup_steps,
        total_training_steps=trainingConfig.train_steps,
        scheduler_type=trainingConfig.scheduler_type,
        min_lr=trainingConfig.min_learning_rate
    )

    set_step_lr(current_actual_lr, optimizer)

    model.train()

    with autocast(device_type=device.type, enabled=usesAmpOrNot(trainingConfig.training_precision), dtype=trainingConfig.precision_dtype):

        with record_function("forward_pass"):
            logits = model(batch_input_ids)

        with record_function("loss_calculation"):
            loss = criterion(logits.view(-1, logits.size(-1)), batch_target_ids.view(-1))

    with record_function("optimizer_zero_grad"):
        optimizer.zero_grad(set_to_none=True)


    if trainingConfig.training_precision == 'float16':
        scaler = trainingConfig.scaler

        with record_function("scaler_backward_pass"):
            scaler.scale(loss).backward()

        with record_function("scaler_optimizer_step"):
            scaler.step(optimizer)

        with record_function("scaler_update"):
            scaler.update()
    else:
        with record_function("backward_pass"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()

    loss_val = loss.item()

    step_log(config=config,
             step_num=step_num,
             loss=loss,
             curr_lr=current_actual_lr,
             wandb=wandb,
             device=device,
             profiler_obj=profiler_obj,
             wandb_run_obj=wandb_run_obj,
             loss_val=loss_val,
             current_actual_lr=current_actual_lr)



def validation_run(model, val_loader, criterion, device, wandb, wandb_run, ENABLE_WANDB, PIN_MEMORY_DATALOADER):
    if ENABLE_WANDB and wandb_run:
        # --- Optional: Validation pass after training ---
        if val_loader:
            print("\nRunning validation...")
            model.eval()
            total_val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device,
                                                      non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)
                    target_ids = batch['labels'].to(device,
                                                    non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)

                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    total_val_loss += loss.item()
                    val_batches += 1
            if val_batches > 0:
                avg_val_loss = total_val_loss / val_batches
                print(f"Average Validation Loss: {avg_val_loss:.4f}")
                wandb.summary["avg_val_loss"] = avg_val_loss
            else:
                print("No batches in validation loader.")
        else:
            print("No validation loader provided.")

        print("Finishing W&B run...")
        wandb.finish()
        print("W&B run finished.")
    elif ENABLE_WANDB and not wandb_run:
        print("W&B enabled but init failed. No W&B run to finish.")
    else:
        print("W&B disabled. No W&B run to finish.")
