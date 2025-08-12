import contextlib
import torch
from torch.profiler import profile, ProfilerActivity

import wandb

from main_space.utils.settings_utils import get_wandb_config, get_training_config, get_dataset_config, get_hyperparameter_config, get_profiler_config

def get_profiler_trace_handler(ENABLE_PROFILER, CUSTOM_TRACE_HANDLER, wandb_run_dir):
    if not ENABLE_PROFILER:
        raise ValueError("In getTraceHandler, ENABLE_PROFILER is False.")

    if CUSTOM_TRACE_HANDLER:
        import os
        trace_dir = os.path.join(wandb_run_dir, "pytorch_traces")  # wandb_run.dir is the sync dir
        os.makedirs(trace_dir, exist_ok=True)
        print(f"PyTorch profiler traces will be saved to: {trace_dir}")

        # This handler will save traces to the specified directory.
        # W&B should then automatically sync files from wandb_run.dir.
        on_trace_ready_handler = torch.profiler.tensorboard_trace_handler(
            dir_name=trace_dir,  # Save to our specified trace_dir
            worker_name=None,  # Usually fine as None for single process
            use_gzip=False  # Easier to inspect if not gzipped initially
        )
        return on_trace_ready_handler
    else:
        return wandb.profiler.torch_trace_handler()


def get_profiler_context(wandb_run, device):

    wandbConfig = get_wandb_config()
    profilerConfig = get_profiler_config()


    print("PyTorch Profiler is ENABLED.")
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    profiler_schedule = torch.profiler.schedule(
        wait=profilerConfig.profiler_wait_steps,
        warmup=profilerConfig.profiler_warmup_steps,
        active=profilerConfig.profiler_active_steps,
        repeat=profilerConfig.profiler_repeat_cycles
    )

    trace_handler = get_profiler_trace_handler(wandbConfig.is_profiler_enabled, wandbConfig.is_trace_handler_custom, wandb_run.dir)

    profiler_context = profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=trace_handler,
        record_shapes=profilerConfig.profiler_record_shapes,
        profile_memory=profilerConfig.profiler_profile_memory,
        with_flops=profilerConfig.profiler_with_ops,
        with_stack=profilerConfig.profiler_with_stack,
    )
    return profiler_context


def setup_wandb_watch(model, wandb_run):
    wandbConfig = get_wandb_config()

    if wandbConfig.is_wandb_enabled and wandb_run and wandbConfig.wandb_watch_level != "none":
        print(f"Setting up W&B model watch (Level: {wandbConfig.wandb_watch_level}, Freq: {wandbConfig.wandb_log_freq_model_watch})...")

        wandb.watch(model, log=wandbConfig.wandb_watch_level, log_freq=wandbConfig.wandb_log_freq_model_watch, log_graph=wandbConfig.does_wandb_log_graph)


def initialize_wandb():


    wandbConfig = get_wandb_config()
    trainingConfig = get_training_config()
    datasetConfig = get_dataset_config()
    hyperParamConfig = get_hyperparameter_config()

    if not wandbConfig.is_wandb_enabled:
        return None


    # TODO: Add number of parameters in model
    print(f"Attempting to initialize W&B (Project: {wandbConfig.wandb_project_name}, Entity: {wandbConfig.wandb_entity})...")
    # TODO: Sigurno se moze ovo napravit da sve parametre iz settingsa se uzmu, ili napravit helper funkciju
    config_dict = {
        "vocab_size": hyperParamConfig.vocab_size,
        "d_model": hyperParamConfig.d_model,
        "num_heads": hyperParamConfig.num_heads,
        "num_layers": hyperParamConfig.num_layers,
        "ctx_len": hyperParamConfig.ctx_len,
        "dropout_rate": hyperParamConfig.dropout_rate,
        "num_iterations": trainingConfig.train_steps,
        "batch_size": trainingConfig.batch_size,
        "learning_rate": trainingConfig.peak_lr,
        "scheduler_type": trainingConfig.scheduler_type,
        "warmup_steps": trainingConfig.warmup_steps,
        "min_lr_cosine": trainingConfig.min_lr,
        "precision": trainingConfig.training_precision,
        "effective_precision": str(trainingConfig.precision_dtype)
    }

    if wandbConfig.is_profiler_enabled:

        profilerConfig = wandbConfig.profilerConfig
        config_dict.update({
            "profiler_wait": profilerConfig.profiler_wait_steps,
            "profiler_warmup": profilerConfig.profiler_warmup_steps,
            "profiler_active": profilerConfig.profiler_active_steps,
            "profiler_repeat": profilerConfig.profiler_repeat_cycles,
            "profiler_record_shapes": profilerConfig.profiler_record_shapes,
            "profiler_profile_memory": profilerConfig.profiler_profile_memory,
            "profiler_with_stack": profilerConfig.profiler_with_stack,
        })

    try:
        wandb_run = wandb.init(
            project=wandbConfig.wandb_project_name,
            entity=wandbConfig.wandb_entity,
            config=config_dict,
            name=hyperParamConfig.run_name,
        )
        if wandb_run:
            print(f"W&B Initialized. Run URL: {wandb_run.url}")
        else:
            raise BrokenPipeError("W&B initialization in log_utils.initialize_wandb call returned None, but no error was raised.")

        return wandb_run
    except Exception as e:
        raise BrokenPipeError(f"Error during W&B initialization in log_utils.initialize_wandb: {e}.")


def print_model_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {num_params:,} trainable parameters.")
    # print(f"FIX print_model_params, CURRENTLY MODEL IS BEING PASSED INTO IT, BECAUSE OTHERWISE IT WOULD"
    #       f"BE A CIRCULAR IMPORT.")

def print_teacher_model_params(teacher_model):
    num_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print(f"Teacher model instantiated with {num_params:,} trainable parameters.")

def log_validation_step(step_num, curr_avg_ce_loss, curr_avg_periodic_losses, device, profiler_obj, wandb_run_obj):

    wandbConfig = get_wandb_config()
    trainingConfig = get_training_config()

    if (step_num + 1) % (wandbConfig.wandb_log_freq_metrics) == 0:
        print(
            f"Step [{step_num + 1}/{trainingConfig.train_steps}], Loss: {curr_avg_ce_loss:.4f}, Bucketed: {curr_avg_periodic_losses}")

    if wandbConfig.is_wandb_enabled and wandb_run_obj and (step_num + 1) % wandbConfig.wandb_log_freq_metrics == 0:
        log_data = {
            "avg_validation_loss": curr_avg_ce_loss,
            "iteration": step_num + 1,
        }

        period_size = 64
        for i, period_loss in enumerate(curr_avg_periodic_losses):
            log_data[f"avg_loss_ctx_{i * period_size}_{(i+1)*period_size-1}"] = period_loss

        if torch.cuda.is_available():
            log_data["gpu_mem_alloc_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            log_data["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)

        wandb.log(log_data, step=step_num + 1)

    # --- Inform the profiler that a step is complete (if profiler is active) ---
    if profiler_obj:
        profiler_obj.step()

def step_log(step_num, ce_only_loss, soft_loss, loss_hidd_total, periodic_losses, curr_lr, device, wandb, wandb_run_obj, current_actual_lr,
             avg_val_loss, total_ops, aux_head_losses):

    wandbConfig = get_wandb_config()
    trainingConfig = get_training_config()

    if (step_num + 1) % (wandbConfig.wandb_log_freq_metrics) == 0:

        print_heads = ""
        if aux_head_losses is not None:
            print_heads = "[" + ", ".join([f"aux_loss_{i}: {aux_loss:.2f}" for i, aux_loss in enumerate(aux_head_losses)]) + "]"
            print("Aux LOSSES: " + str(print_heads))
            print_heads = f", Aux losses: {print_heads}"

        print(
            f"Step [{step_num + 1}/{trainingConfig.train_steps}],"
            f" Hard loss: {ce_only_loss:.4f},"
            f"{(' Avg val loss: ' + str(avg_val_loss) + ' ') if avg_val_loss is not None else ''}"
            f"LR: {current_actual_lr:.2e}, Total ops: {total_ops:.2e}"
            f"{print_heads}")

    if wandbConfig.is_wandb_enabled and wandb_run_obj and (step_num + 1) % wandbConfig.wandb_log_freq_metrics == 0:
        log_data = {
            "train_loss": ce_only_loss,
            "iteration": step_num + 1,
            "learning_rate": curr_lr,
            "total_ops": total_ops,

        }
        if aux_head_losses is not None:
            for i, aux_loss in enumerate(aux_head_losses):
                log_data[f"aux_loss_{i}"] = aux_loss

        if avg_val_loss:
            log_data["avg_loss_val"] = avg_val_loss

        if soft_loss:
            log_data["soft_loss"] = soft_loss

        if loss_hidd_total:
            log_data["loss_hidd_total"] = loss_hidd_total

        if periodic_losses is not None:
            period_size = 64
            for i, period_loss in enumerate(periodic_losses):
                log_data[f"loss_ctx_{i * period_size}_{(i+1)*period_size-1}"] = period_loss.item()

        if trainingConfig.scaler:
            log_data["grad_scaler_scale"] = trainingConfig.scaler.get_scale()

        if torch.cuda.is_available():
            log_data["gpu_mem_alloc_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            log_data["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)

        wandb.log(log_data, step=step_num + 1)


