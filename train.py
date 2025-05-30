import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import wandb

from model import MyTransformerLM

# --- SCRIPT SETTINGS ---

# --- Overall Logging Control ---
ENABLE_WANDB = True       # Master switch for all W&B interactions
ENABLE_PROFILER = False    # Master switch for torch.profiler
CUSTOM_TRACE_HANDLER = False

# W&B Configuration
WANDB_PROJECT_NAME = "self-distill-research"
WANDB_ENTITY = "luka_newbie"
WANDB_WATCH_LEVEL = "all"    # Options: "all", "gradients", "parameters", "none"
WANDB_LOG_FREQ_MODEL_WATCH = 100 # Frequency for wandb.watch (if not "none") (e.g., every 100 steps)
WANDB_LOG_FREQ_METRICS = 10    # Frequency for wandb.log() for loss, etc. (e.g., every 10 steps)
WANDB_LOG_GRAPH = True         # For wandb.watch(), log the model graph

# --- PyTorch Profiler Specific Configuration ---
# (Only used if ENABLE_PROFILER is True and ENABLE_WANDB is True)
PROFILER_WAIT_STEPS = 2
PROFILER_WARMUP_STEPS = 2
PROFILER_ACTIVE_STEPS = 2
PROFILER_REPEAT_CYCLES = 1      # Total active steps = PROFILER_ACTIVE_STEPS * PROFILER_REPEAT_CYCLES
PROFILER_RECORD_SHAPES = True
PROFILER_PROFILE_MEMORY = True  # Can be True/False
PROFILER_WITH_STACK = False     # Set to False by default, as it can be costly

# Model Configuration (passed to MyTransformerLM)
VOCAB_SIZE = 1000  # Dummy vocab size
D_MODEL = 128      # Embedding dimension / model dimension
NUM_HEADS = 4      # Number of attention heads
NUM_LAYERS = 2     # Number of Transformer blocks
CTX_LEN = 64   # Max sequence length for dummy data and positional embeddings
DROPOUT_RATE = 0.1

# Training Configuration
NUM_ITERATIONS = 102
BATCH_SIZE = 8
LEARNING_RATE = 1e-4


# --- Configuration Validation ---
def validate_config():
    print("Validating configuration...")
    if ENABLE_WANDB and wandb is None:
        raise ImportError(
            "W&B is enabled (ENABLE_WANDB=True) but the 'wandb' library is not installed. Please install it: pip install wandb")

    if ENABLE_PROFILER and not ENABLE_WANDB:
        raise ValueError("Configuration Error: ENABLE_PROFILER is True, but ENABLE_WANDB is False. "
                         "The current profiler setup relies on W&B for trace handling. "
                         "If you want to use the profiler, ENABLE_WANDB must also be True.")

    if ENABLE_PROFILER and wandb is not None and not hasattr(wandb.profiler, 'torch_trace_handler'):
        # This check might be too strict if older wandb versions have different paths
        # but good for ensuring the expected handler exists.
        print("Warning: ENABLE_PROFILER is True, but `wandb.profiler.torch_trace_handler` might not be available. "
              "Ensure your W&B library is up-to-date. Profiling might not work as expected.")

    valid_watch_levels = ["all", "gradients", "parameters", "none"]
    if WANDB_WATCH_LEVEL not in valid_watch_levels:
        raise ValueError(
            f"Configuration Error: WANDB_WATCH_LEVEL must be one of {valid_watch_levels}, got '{WANDB_WATCH_LEVEL}'.")

    print("Configuration appears valid.")


def train_step_logic(step_num, model, criterion, optimizer, device, current_batch_size,
                     log_to_wandb_flag, wandb_run_obj, profiler_obj=None):
    model.train()

    # --- Dummy Data Generation ---
    with record_function("data_generation"):
        input_ids = torch.randint(0, VOCAB_SIZE, (current_batch_size, CTX_LEN), device=device)
        target_ids = torch.randint(0, VOCAB_SIZE, (current_batch_size, CTX_LEN), device=device)

    # --- Forward Pass ---
    with record_function("forward_pass"):
        logits = model(input_ids)  # Shape: [BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE]

    # --- Loss Calculation ---
    with record_function("loss_calculation"):
        loss = criterion(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

    # --- Backward Pass & Optimization ---
    with record_function("optimizer_zero_grad"):
        optimizer.zero_grad(set_to_none=True)  # Common optimization
    with record_function("backward_pass"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()

    # --- W&B Logging (if enabled and it's a logging step) ---
    if log_to_wandb_flag and wandb_run_obj and (step_num + 1) % WANDB_LOG_FREQ_METRICS == 0:
        log_data = {
            "train_loss": loss.item(),
            "iteration": step_num + 1,
            "learning_rate": LEARNING_RATE  # Fixed for now
        }
        if torch.cuda.is_available():
            log_data["gpu_mem_alloc_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            log_data["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        wandb.log(log_data, step=step_num + 1)

    # --- Inform the profiler that a step is complete (if profiler is active) ---
    if profiler_obj:
        profiler_obj.step()

    return loss.item()


def getTraceHandler(wandb_run_dir):

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



def main():
    validate_config()  # Perform configuration checks first

    wandb_run = None  # Initialize wandb run object

    # --- 1. W&B Initialization (if enabled) ---
    if ENABLE_WANDB:
        print(f"Attempting to initialize W&B (Project: {WANDB_PROJECT_NAME}, Entity: {WANDB_ENTITY})...")
        config_dict = {
            # Model params
            "vocab_size": VOCAB_SIZE, "d_model": D_MODEL, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "max_seq_len": CTX_LEN, "dropout_rate": DROPOUT_RATE,
            # Training params
            "num_iterations": NUM_ITERATIONS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            # Logging params
            "enable_profiler": ENABLE_PROFILER, "wandb_watch_level": WANDB_WATCH_LEVEL,
            "log_freq_metrics": WANDB_LOG_FREQ_METRICS, "log_freq_model_watch": WANDB_LOG_FREQ_MODEL_WATCH,
        }
        if ENABLE_PROFILER:  # Add profiler specific configs if it's enabled
            config_dict.update({
                "profiler_wait": PROFILER_WAIT_STEPS, "profiler_warmup": PROFILER_WARMUP_STEPS,
                "profiler_active": PROFILER_ACTIVE_STEPS, "profiler_repeat": PROFILER_REPEAT_CYCLES,
                "profiler_record_shapes": PROFILER_RECORD_SHAPES,
                "profiler_profile_memory": PROFILER_PROFILE_MEMORY,
                "profiler_with_stack": PROFILER_WITH_STACK,
            })

        try:
            wandb_run = wandb.init(
                project=WANDB_PROJECT_NAME,
                entity=WANDB_ENTITY,
                config=config_dict
            )
            if wandb_run:
                print(f"W&B Initialized. Run URL: {wandb_run.url}")
            else:  # Should ideally not happen if wandb.init doesn't raise error
                print("W&B initialization call returned None, but no error was raised. W&B features might be limited.")
        except Exception as e:
            print(f"Error initializing W&B: {e}. W&B features will be disabled.")
            # Effectively disable W&B for the rest of the script if init fails
            # This might be redundant if wandb_run remains None, but good for clarity.
            # ENABLE_WANDB = False # Let's not modify global config inside main.
            # Instead, subsequent checks for `wandb_run` will handle this.
            wandb_run = None  # Ensure it's None if init failed
    else:
        print("W&B is DISABLED by configuration.")

    # --- 2. Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Model Instantiation ---
    model = MyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=NUM_HEADS,
        n_layers=NUM_LAYERS,
        ctx_size=CTX_LEN,
        p_dropout=DROPOUT_RATE
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {num_params:,} trainable parameters.")
    if ENABLE_WANDB and wandb_run:
        wandb.summary["model_parameters"] = num_params

    # --- 4. W&B Watch (if enabled and level is not "none") ---
    if ENABLE_WANDB and wandb_run and WANDB_WATCH_LEVEL != "none":
        print(f"Setting up W&B model watch (Level: {WANDB_WATCH_LEVEL}, Freq: {WANDB_LOG_FREQ_MODEL_WATCH})...")
        wandb.watch(model, log=WANDB_WATCH_LEVEL, log_freq=WANDB_LOG_FREQ_MODEL_WATCH, log_graph=WANDB_LOG_GRAPH)

    # --- 5. Optimizer and Loss Criterion Setup ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting dummy training for {NUM_ITERATIONS} iterations...")

    # --- 6. Training Loop (Profiler-aware or standard) ---
    if ENABLE_PROFILER and ENABLE_WANDB and wandb_run:  # Profiler active (implies W&B is also active and init succeeded)
        print("PyTorch Profiler with W&B Handler is ENABLED.")
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profiler_schedule = torch.profiler.schedule(
            wait=PROFILER_WAIT_STEPS,
            warmup=PROFILER_WARMUP_STEPS,
            active=PROFILER_ACTIVE_STEPS,
            repeat=PROFILER_REPEAT_CYCLES
        )

        trace_handler = getTraceHandler(wandb_run.dir)

        with profile(
                activities=activities,
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                # Assumes wandb & wandb_run are valid due to checks
                record_shapes=PROFILER_RECORD_SHAPES,
                profile_memory=PROFILER_PROFILE_MEMORY,
                with_stack=PROFILER_WITH_STACK
        ) as prof:
            for step in range(NUM_ITERATIONS):
                loss_val = train_step_logic(step, model, criterion, optimizer, device, BATCH_SIZE,
                                            ENABLE_WANDB, wandb_run, profiler_obj=prof)
                if (step + 1) % (WANDB_LOG_FREQ_METRICS * 20) == 0:  # Console print less frequently
                    print(f"Step [{step + 1}/{NUM_ITERATIONS}], Loss: {loss_val:.4f} (Profiler Active)")
    else:  # Profiler is disabled, or W&B is disabled/failed
        if ENABLE_PROFILER and (not ENABLE_WANDB or not wandb_run):
            print(
                "Profiler was configured to be enabled, but W&B is not available/initialized for trace handling. Running without profiler.")
        else:
            print("Profiler is DISABLED by configuration. Running standard training loop.")

        for step in range(NUM_ITERATIONS):
            loss_val = train_step_logic(step, model, criterion, optimizer, device, BATCH_SIZE,
                                        ENABLE_WANDB, wandb_run, profiler_obj=None)
            if (step + 1) % (WANDB_LOG_FREQ_METRICS * 20) == 0:  # Console print less frequently
                print(f"Step [{step + 1}/{NUM_ITERATIONS}], Loss: {loss_val:.4f}")

    print("\nDummy training completed.")

    # --- 7. W&B Finish (if enabled and initialized) ---
    if ENABLE_WANDB and wandb_run:
        print("Finishing W&B run...")
        wandb.finish()
        print("W&B run finished.")
    elif ENABLE_WANDB and not wandb_run:
        print("W&B was enabled in config, but initialization failed or was skipped. No W&B run to finish.")
    else:  # ENABLE_WANDB is False
        print("W&B was disabled. No W&B run to finish.")


if __name__ == "__main__":
    # IMPORTANT: Set your WANDB_ENTITY before running if WANDB is enabled!
    if ENABLE_WANDB and (
            WANDB_ENTITY == "your_wandb_username_or_team" or WANDB_ENTITY == "" or WANDB_ENTITY == "luka_newbie"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! W&B IS ENABLED! PLEASE VERIFY 'WANDB_ENTITY' IS SET CORRECTLY TO   !!!")
        print(f"!!! YOUR W&B USERNAME OR TEAM NAME. CURRENTLY: '{WANDB_ENTITY}'             !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    try:
        main()
    except ValueError as e:  # Catch config validation errors
        print(f"CONFIGURATION ERROR: {e}")
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # For debugging, you might want to re-raise or print traceback
        import traceback

        traceback.print_exc()