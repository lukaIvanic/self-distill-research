import torch
from torch.amp import autocast
from torch.nn import functional as F
from torch import nn

from torch.profiler import record_function

import wandb

from main_space.utils.log_utils import step_log
from main_space.utils.hyperparameter_utils import usesAmpOrNot, calculate_lr, calculate_distill_alpha
from main_space.utils.settings_utils import get_training_config, get_distill_config


def set_step_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_loss_classic(model, criterion, batch_input_ids, batch_target_ids):
    with record_function("forward_pass"):
        logits = model(batch_input_ids)

    with record_function("loss_calculation"):
        per_token_losses = criterion(logits.view(-1, logits.size(-1)), batch_target_ids.view(-1))

        overall_loss = per_token_losses.mean()

        losses_by_position = per_token_losses.view(batch_input_ids.size(0), -1)

        avg_loss_per_position = losses_by_position.mean(dim=0)

        period = 64
        periodic_losses = avg_loss_per_position.view(-1, period).mean(dim=1)

    return overall_loss, periodic_losses


def get_loss_soft(step_num, model, teacher_model, criterion, batch_input_ids, batch_target_ids):
    with record_function("forward_pass_teacher"):
        with torch.no_grad(): # TODO: does this actually do anything?
            teacher_logits = teacher_model(batch_input_ids)

    with record_function("forward_pass_student"):
        student_logits = model(batch_input_ids)

    with record_function("loss_calculation_hard_labels"):
        L_hard = criterion(student_logits.view(-1, student_logits.size(-1)), batch_target_ids.view(-1)).mean()

        if step_num % 400 == 0 or step_num == 0:
            l_hard_item = L_hard.item()
            print(f"hard_loss_item: {l_hard_item}")


    with record_function("loss_calculation_soft_labels"):
        temperature = 2.0

        distillation_loss_fn = nn.KLDivLoss(reduction='none')

        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

        teacher_probs = F.softmax(teacher_logits_flat / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits_flat / temperature, dim=-1)

        L_soft = distillation_loss_fn(
            input=student_log_probs,
            target=teacher_probs
        ).sum(dim=1).mean()

        L_soft = L_soft * (temperature * temperature) # compensating for 1/T^2 bias

        if step_num % 400 == 0 or step_num == 0:
            l_soft_item = L_soft.item()
            print(f"soft_loss_item: {l_soft_item}")



    ratio = L_hard.item() / L_soft.item()

    L_soft *= ratio

    alpha = 0.5
    L = alpha * L_hard + (1 - alpha) * L_soft



    return L, L_hard, L_soft


def get_loss_hidd_distill(step_num, model, teacher_model, criterion, batch_input_ids, batch_target_ids):

    with record_function("forward_pass_teacher"):
        with torch.no_grad():
            _, hidd_states_teacher = teacher_model.forward_with_out_hidd_for_distill(batch_input_ids)

    with record_function("forward_pass_student"):
        student_logits, hidd_states_student = model.forward_with_out_hidd_for_distill(batch_input_ids)

    with record_function("loss_calculation_hard_labels"):
        L_hard = criterion(student_logits.view(-1, student_logits.size(-1)), batch_target_ids.view(-1)).mean()

        if step_num % 100 == 0 or step_num == 0:
            l_hard_item = L_hard.item()
            print(f"hard_loss_item: {l_hard_item}")



    with record_function("loss_calculation_hidd_distill"):

        L_hidd_total = 0.0

        teacher_len = len(hidd_states_teacher)
        student_len = len(hidd_states_student)

        curr_indx = teacher_len -1
        indexes = [curr_indx]
        steps = (teacher_len) // (student_len-1)

        for i in range(curr_indx-1, -1, -steps):
            indexes.insert(0, i)


        for i in range(len(hidd_states_student)):
            teacher_state = hidd_states_teacher[indexes[i]]
            student_state = hidd_states_student[i]

            if teacher_state.shape != student_state.shape:
                raise ValueError(f"Teacher ({teacher_state.shape}) and student ({student_state.shape}) "
                                 "hidden states must have the same shape for MSE distillation.")

            L_hidd = F.mse_loss(
                input=student_state,
                target=teacher_state,
                reduction='mean'
            )

            if step_num % 100 == 0 or step_num == 0:
                l_hidd_item = L_hidd.item()
                print(f"l_hidd_{i}_item: {l_hidd_item}")

            L_hidd_total += L_hidd


    ratio = L_hard.item() / L_hidd_total.item()

    L_hidd_total *= ratio
    alpha = 0.5
    L = alpha * L_hard + (1 - alpha) * L_hidd_total

    return L, L_hard, L_hidd_total


def get_loss_attn_distill(step_num, model, criterion, batch_input_ids, batch_target_ids):
    distillConfig = get_distill_config()

    with record_function("forward_pass_distill"):
        logits, attns_per_block = model.forward_with_attn_for_distill(batch_input_ids)

    with record_function("loss_calculation_distill"):
        ce_loss = criterion(logits.view(-1, logits.size(-1)), batch_target_ids.view(-1)).mean()

        ce_item = ce_loss.item()
        print(f"ce_item: {ce_item}")

        teacher_attns = attns_per_block[distillConfig.teacher_index]
        student_attns = attns_per_block[distillConfig.student_index]

        B, H, T_q, T_k = teacher_attns.shape

        student_attns_logged = (student_attns + 1e-8).log()
        teacher_attns_logged = (teacher_attns + 1e-8).log()

        student_probs_for_kl = student_attns.view(B * H * T_q, T_k)
        student_log_probs_for_kl = student_attns_logged.view(B * H * T_q, T_k)

        teacher_probs_for_kl = teacher_attns.view(B * H * T_q, T_k)
        teacher_log_probs_for_kl = teacher_attns_logged.view(B * H * T_q, T_k)

        kl_loss = F.kl_div(
            input=student_log_probs_for_kl,
            target=teacher_log_probs_for_kl,
            reduction='batchmean',
            log_target=True
        )

        kl_item = kl_loss.item()
        print(f"kl_item: {kl_item}")

        step_distill_alpha = calculate_distill_alpha(step_num)
        total_loss = ce_loss + step_distill_alpha * kl_loss

    return total_loss, ce_loss


def clip_gradients(model, optimizer):
    trainingConfig = get_training_config()

    if not trainingConfig.doesClipGradients:
        return

    # TODO: add elsewhere initial validation check for gradient norm value setting

    if trainingConfig.scaler is not None:
        with record_function("scaler_unscale_gradients"):
            trainingConfig.scaler.unscale_(optimizer)


    # TODO: implement logging for gradient_clipping before, and after, as a general checker.
    with record_function("gradient_clipping"):
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=trainingConfig.gradient_clip_norm,
            norm_type=2.0,
            error_if_nonfinite=True
        )
logging_loss_accumulator = {
    'ce_only_loss': 0.0,
    'loss_soft': 0.0,
    'loss_hidd_total': 0.0,
}

def make_train_step(step_num,
                    model,
                    teacher_model,
                    criterion,
                    optimizer,
                    device,
                    batch_input_ids,
                    batch_target_ids,
                    wandb_run_obj):
    global logging_loss_accumulator

    trainingConfig = get_training_config()

    accumulation_steps = 1
    actual_step_num = step_num // accumulation_steps

    current_actual_lr = calculate_lr(step_num=actual_step_num)
    set_step_lr(current_actual_lr, optimizer)

    model.train()

    periodic_losses = None
    loss_soft = None
    loss_hidd_total = None




    with autocast(device_type=device.type, enabled=usesAmpOrNot(trainingConfig.training_precision),
                  dtype=trainingConfig.precision_dtype):

        if trainingConfig.distill_enabled:

            if trainingConfig.distillConfig.distill_mode == 'logits_outputs':
                loss, ce_only_loss, loss_soft = get_loss_soft(actual_step_num,
                                                   model,
                                                   teacher_model,
                                                   criterion,
                                                   batch_input_ids,
                                                   batch_target_ids)

            elif trainingConfig.distillConfig.distill_mode == 'hidd_distill':
                loss, ce_only_loss, loss_hidd_total = get_loss_hidd_distill(actual_step_num,
                                                           model,
                                                           teacher_model,
                                                           criterion,
                                                           batch_input_ids,
                                                           batch_target_ids)
            else:
                raise BrokenPipeError(f"distill_mode should be 'logits_output', but was {trainingConfig.distillConfig.distill_mode}.")



        else:
            loss, periodic_losses = get_loss_classic(model, criterion, batch_input_ids, batch_target_ids)
            ce_only_loss = loss

    loss /= accumulation_steps
    with record_function("backward_pass"):
        loss.backward()




    logging_loss_accumulator["ce_only_loss"] += ce_only_loss.item()
    if loss_soft:
        logging_loss_accumulator["loss_soft"] += (loss_soft / accumulation_steps).item()
    if loss_hidd_total:
        logging_loss_accumulator["loss_hidd_total"] += (loss_hidd_total / accumulation_steps).item()



    if (step_num + 1) % accumulation_steps == 0:

        with record_function("gradient_clipping"):
            clip_gradients(model, optimizer)

        with record_function("optimizer_step"):
            optimizer.step()

        step_log(step_num=actual_step_num,
                 ce_only_loss=logging_loss_accumulator["ce_only_loss"],
                 soft_loss=logging_loss_accumulator["loss_soft"],
                 loss_hidd_total=logging_loss_accumulator["loss_hidd_total"],
                 periodic_losses=periodic_losses,
                 curr_lr=current_actual_lr,
                 wandb=wandb,
                 device=device,
                 wandb_run_obj=wandb_run_obj,
                 current_actual_lr=current_actual_lr)

        with record_function("optimizer_zero_grad"):
            optimizer.zero_grad(set_to_none=True)

        logging_loss_accumulator["ce_only_loss"] = 0.0
        logging_loss_accumulator["loss_soft"] = 0.0
        logging_loss_accumulator["loss_hidd_total"] = 0.0


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
