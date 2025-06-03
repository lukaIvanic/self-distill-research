import contextlib
import torch
from torch.profiler import profile, ProfilerActivity

import wandb



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


def get_profiler_or_null_context(ENABLE_PROFILER, ENABLE_WANDB, wandb_run, device, wandbConfig):
    profilerConfig = wandbConfig.profilerConfig

    if not (ENABLE_PROFILER and ENABLE_WANDB and wandb_run):
        return contextlib.nullcontext()

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

    trace_handler = get_profiler_trace_handler(ENABLE_PROFILER, wandbConfig.is_trace_handler_custom, wandb_run.dir)

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


def enable_wandb_watch(config, model, wandb_run):
    wandbConfig = config.wandbConfig

    if wandbConfig.is_wandb_enabled and wandb_run and wandbConfig.wandb_watch_level != "none":
        print(f"Setting up W&B model watch (Level: {wandbConfig.wandb_watch_level}, Freq: {wandbConfig.wandb_log_freq_model_watch})...")

        wandb.watch(model, log=wandbConfig.wandb_watch_level, log_freq=wandbConfig.wandb_log_freq_model_watch, log_graph=wandbConfig.does_wandb_log_graph)


def initialize_wandb(config):


    wandbConfig = config.wandbConfig
    trainingConfig = config.trainingConfig
    datasetConfig = config.datasetConfig

    if not wandbConfig.is_wandb_enabled:
        return None

    hyperParamConfig = trainingConfig.hyperParamConfig

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
        "seed": trainingConfig.seed,
        "dataset_name": datasetConfig.dataset_name,
        "dataset_config": datasetConfig.dataset_config,
        "enable_profiler": wandbConfig.is_profiler_enabled,
        "wandb_watch_level": wandbConfig.wandb_watch_level,
        "log_freq_metrics": wandbConfig.wandb_log_freq_metrics,
        "log_freq_model_watch": wandbConfig.wandb_log_freq_model_watch,
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
            config=config_dict
        )
        if wandb_run:
            print(f"W&B Initialized. Run URL: {wandb_run.url}")
        else:  # Should ideally not happen if wandb.init doesn't raise error
            print("W&B initialization call returned None, but no error was raised. W&B features might be limited.")

        return wandb_run
    except Exception as e:
        print(f"Error initializing W&B: {e}. W&B features will be disabled.")


def step_log(config, step_num, ce_only_loss, curr_lr, device, wandb, profiler_obj, wandb_run_obj, loss_val, current_actual_lr):

    wandbConfig = config.wandbConfig
    trainingConfig = config.trainingConfig

    if (step_num + 1) % (wandbConfig.wandb_log_freq_metrics) == 0:
        print(
            f"Step [{step_num + 1}/{trainingConfig.train_steps}], Loss: {loss_val:.4f}, LR: {current_actual_lr:.2e}")

    if wandbConfig.is_wandb_enabled and wandb_run_obj and (step_num + 1) % wandbConfig.wandb_log_freq_metrics == 0:
        log_data = {
            "train_loss": ce_only_loss.item(),
            "iteration": step_num + 1,
            "learning_rate": curr_lr
        }

        if trainingConfig.scaler:
            log_data["grad_scaler_scale"] = trainingConfig.scaler.get_scale()

        if torch.cuda.is_available():
            log_data["gpu_mem_alloc_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            log_data["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)

        wandb.log(log_data, step=step_num + 1)

    # --- Inform the profiler that a step is complete (if profiler is active) ---
    if profiler_obj:
        profiler_obj.step()
