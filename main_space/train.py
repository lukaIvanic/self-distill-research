import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import wandb

from torch.utils.data import DataLoader
from tokenizers import Tokenizer  # For loading tokenizer to get special IDs

from model import MyTransformerLM
from data_management import load_and_process_dataset_for_lm, CausalLMTrainingDataset

# --- SCRIPT SETTINGS ---

# --- Overall Logging Control ---
ENABLE_WANDB = True       # Master switch for all W&B interactions
ENABLE_PROFILER = True    # Master switch for torch.profiler
CUSTOM_TRACE_HANDLER = True

# --- W&B Configuration ---
WANDB_PROJECT_NAME = "self-distill-research"
WANDB_ENTITY = "luka_newbie"
WANDB_WATCH_LEVEL = "all"    # Options: "all", "gradients", "parameters", "none"
WANDB_LOG_FREQ_MODEL_WATCH = 100 # Frequency for wandb.watch (if not "none") (e.g., every 100 steps)
WANDB_LOG_FREQ_METRICS = 10    # Frequency for wandb.log() for loss, etc. (e.g., every 10 steps)
WANDB_LOG_GRAPH = True         # For wandb.watch(), log the model graph

# --- PyTorch Profiler Specific Configuration ---
# (Only used if ENABLE_PROFILER is True and ENABLE_WANDB is True)
PROFILER_WAIT_STEPS = 200
PROFILER_WARMUP_STEPS = 5
PROFILER_ACTIVE_STEPS = 5
PROFILER_REPEAT_CYCLES = 5      # Total active steps = PROFILER_ACTIVE_STEPS * PROFILER_REPEAT_CYCLES
PROFILER_RECORD_SHAPES = True
PROFILER_PROFILE_MEMORY = True  # Can be True/False
PROFILER_WITH_STACK = False     # Set to False by default, as it can be costly

# Model Configuration (passed to MyTransformerLM)
VOCAB_SIZE = 5000  # Dummy vocab size
D_MODEL = 512      # Embedding dimension / model dimension
NUM_HEADS = 16      # Number of attention heads
NUM_LAYERS = 8     # Number of Transformer blocks
CTX_LEN = 128   # Max sequence length for dummy data and positional embeddings
DROPOUT_RATE = 0.1

# Training Configuration
NUM_ITERATIONS = int(2e5)
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SEED = 42  # For reproducibility of data shuffling and other random ops

# --- NEW: Data Configuration ---
TOKENIZER_PATH = os.path.join(os.getcwd(),"dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")  # IMPORTANT: UPDATE THIS
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"  # Standard processed version
# Cache directory for Hugging Face datasets (raw and processed by load_and_process_dataset_for_lm)
CACHE_DIR = os.path.join(os.getcwd(),"/dataset_creation/temp_files/cache_hf_datasets")
# DataLoader options
NUM_WORKERS_DATALOADER = 0  # 0 for main process, >0 for multiprocessing. Start with 0 for simplicity/debugging.
PIN_MEMORY_DATALOADER = True
VOCAB_SIZE_FROM_TOKENIZER = True


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


    if not os.path.exists(TOKENIZER_PATH):
        # Try to give a more helpful message if it's the default path
        if TOKENIZER_PATH == "path/to/your/tokenizer.json":
            error_msg = (f"ERROR: Tokenizer not found at default placeholder path: '{TOKENIZER_PATH}'. "
                         "Please update TOKENIZER_PATH in the script settings.")
        else:
            error_msg = f"ERROR: Tokenizer not found at: '{TOKENIZER_PATH}'."
        raise FileNotFoundError(error_msg)

    print("Configuration appears valid.")


# --- Modified train_step_logic to accept data batch ---
def train_step_logic(step_num, model, criterion, optimizer, device,
                     batch_input_ids, batch_target_ids,  # Directly supplied
                     log_to_wandb_flag, wandb_run_obj, profiler_obj=None):
    model.train()

    # batch_input_ids and batch_target_ids are already on the correct device from DataLoader if pin_memory=True
    # Or need to be moved if not:
    # batch_input_ids = batch_input_ids.to(device)
    # batch_target_ids = batch_target_ids.to(device)

    # --- Forward Pass ---
    with record_function("forward_pass"):
        logits = model(batch_input_ids)

    # --- Loss Calculation ---
    with record_function("loss_calculation"):
        loss = criterion(logits.view(-1, logits.size(-1)), batch_target_ids.view(-1))

    # --- Backward Pass & Optimization ---
    with record_function("optimizer_zero_grad"):
        optimizer.zero_grad(set_to_none=True)
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




# --- NEW: Function to get DataLoaders ---
def get_dataloaders(tokenizer_path, ctx_len, dataset_name, dataset_config, cache_dir,
                    batch_size, seed, num_workers, pin_memory):
    print(f"Loading tokenizer from: {tokenizer_path} for DataLoader setup...")
    # Tokenizer is loaded here primarily to get special token IDs for data processing function
    loaded_tokenizer = Tokenizer.from_file(tokenizer_path)
    bos_token_id = loaded_tokenizer.token_to_id("[SEP]")
    eos_token_id = loaded_tokenizer.token_to_id("[CLS]")
    pad_token_id = loaded_tokenizer.token_to_id("[PAD]")

    if None in [bos_token_id, eos_token_id, pad_token_id]:
        raise ValueError("Ensure tokenizer has <s>, </s>, and <pad> tokens defined.")

    actual_vocab_size = loaded_tokenizer.get_vocab_size()
    print(f"Tokenizer vocabulary size: {actual_vocab_size}")

    print("Calling load_and_process_dataset_for_lm...")
    lm_ready_hf_datasets, _ = load_and_process_dataset_for_lm(
        tokenizer_path=tokenizer_path,  # Passed again for consistency, though loaded_tokenizer could be used
        max_seq_len=ctx_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        cache_dir=cache_dir,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )

    pytorch_datasets = {}
    dataloaders = {}
    generator = torch.Generator().manual_seed(seed)  # For reproducible shuffling

    for split_name, hf_split_data in lm_ready_hf_datasets.items():
        if not hf_split_data or len(hf_split_data) == 0:
            print(f"Split '{split_name}' is empty or not found after processing. DataLoader will be None.")
            dataloaders[split_name] = None
            continue

        pytorch_datasets[split_name] = CausalLMTrainingDataset(
            hf_dataset_split=hf_split_data,
            max_seq_len=ctx_len,
            pad_token_id=pad_token_id
        )
        is_train_split = (split_name == 'train')
        dataloaders[split_name] = DataLoader(
            pytorch_datasets[split_name],
            batch_size=batch_size,
            shuffle=is_train_split,
            generator=generator if is_train_split else None,
            num_workers=num_workers,
            pin_memory=pin_memory if torch.cuda.is_available() else False
        )
    # Return tokenizer's actual vocab size along with data loaders
    return dataloaders.get('train'), dataloaders.get('validation'), dataloaders.get('test'), actual_vocab_size


def get_profiler_trace_handler(wandb_run_dir):

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
    validate_config()
    torch.manual_seed(SEED)  # Global seed for other torch ops if any

    wandb_run = None
    final_vocab_size = None  # Will be updated by get_dataloaders

    if ENABLE_WANDB:
        # Initialize W&B (config dict will be prepared after vocab_size is known)
        pass  # Defer init until vocab_size is known

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoaders Setup ---
    print("Setting up DataLoaders...")
    train_loader, val_loader, _test_loader, tokenizer_vocab_size = get_dataloaders(
        tokenizer_path=TOKENIZER_PATH,
        ctx_len=CTX_LEN,
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        cache_dir=CACHE_DIR,
        batch_size=BATCH_SIZE,
        seed=SEED,
        num_workers=NUM_WORKERS_DATALOADER,
        pin_memory=PIN_MEMORY_DATALOADER
    )

    if not train_loader:
        print("ERROR: Training DataLoader could not be created. Exiting.")
        return

    if VOCAB_SIZE_FROM_TOKENIZER:
        final_vocab_size = tokenizer_vocab_size
        print(f"Using vocabulary size from tokenizer: {final_vocab_size}")
    else:
        # This global VOCAB_SIZE is a fallback, not currently used if VOCAB_SIZE_FROM_TOKENIZER is True
        # final_vocab_size = VOCAB_SIZE
        # print(f"Using fixed vocabulary size from script: {final_vocab_size}")
        # For this integration, we mandate using tokenizer's vocab size.
        final_vocab_size = tokenizer_vocab_size
        print(
            f"WARNING: VOCAB_SIZE_FROM_TOKENIZER is False, but we will use tokenizer's vocab size: {final_vocab_size}")

    # --- Now Initialize W&B (if enabled), with actual vocab_size in config ---
    if ENABLE_WANDB:
        print(f"Attempting to initialize W&B (Project: {WANDB_PROJECT_NAME}, Entity: {WANDB_ENTITY})...")
        config_dict = {
            "vocab_size": final_vocab_size, "d_model": D_MODEL, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "ctx_len": CTX_LEN, "dropout_rate": DROPOUT_RATE,
            "num_iterations": NUM_ITERATIONS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "seed": SEED, "dataset_name": DATASET_NAME, "dataset_config": DATASET_CONFIG,
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



    # --- 3. Model Instantiation ---
    model = MyTransformerLM(
        vocab_size=final_vocab_size,  # Use actual vocab size
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
    # For LM, CrossEntropyLoss ignores index -100 by default, which our DataLoader uses for label padding.
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting training for {NUM_ITERATIONS} iterations...")
    print(f"Train loader has ~{len(train_loader)} batches of size {BATCH_SIZE}.")

    train_iter = iter(train_loader)
    current_step = 0

    # --- Training Loop ---
    if ENABLE_PROFILER and ENABLE_WANDB and wandb_run:
        print("PyTorch Profiler is ENABLED.")
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profiler_schedule = torch.profiler.schedule(
            wait=PROFILER_WAIT_STEPS,
            warmup=PROFILER_WARMUP_STEPS,
            active=PROFILER_ACTIVE_STEPS,
            repeat=PROFILER_REPEAT_CYCLES
        )

        trace_handler = get_profiler_trace_handler(wandb_run.dir)

        with profile(
                activities=activities,
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                # Assumes wandb & wandb_run are valid due to checks
                record_shapes=PROFILER_RECORD_SHAPES,
                profile_memory=PROFILER_PROFILE_MEMORY,
                with_stack=PROFILER_WITH_STACK
        ) as prof:
            for step_num in range(NUM_ITERATIONS):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    print(f"Epoch finished at step {step_num}. Resetting train_loader for continued iteration.")
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                input_ids = batch['input_ids'].to(device,
                                                  non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)
                target_ids = batch['labels'].to(device,
                                                non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)



                loss_val = train_step_logic(
                    step_num, model, criterion, optimizer, device,
                    input_ids, target_ids,
                    ENABLE_WANDB, wandb_run, profiler_obj=prof)
                current_step += 1
                if (step_num + 1) % (WANDB_LOG_FREQ_METRICS * 20) == 0:
                    print(f"Step [{step_num + 1}/{NUM_ITERATIONS}], Loss: {loss_val:.4f} (Profiler Active)")
    else:  # Profiler disabled or W&B issue
        if ENABLE_PROFILER:
            print("Profiler was enabled but conditions not met (e.g. W&B disabled). Running without profiler.")
        else:
            print("Profiler is DISABLED. Running standard training loop.")

        for step_num in range(NUM_ITERATIONS):
            try:
                batch = next(train_iter)
            except StopIteration:
                print(f"Epoch finished at step {step_num}. Resetting train_loader for continued iteration.")
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(device,
                                              non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)
            target_ids = batch['labels'].to(device,
                                            non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)

            loss_val = train_step_logic(
                step_num, model, criterion, optimizer, device,
                input_ids, target_ids,
                ENABLE_WANDB, wandb_run, profiler_obj=None)
            current_step += 1
            if (step_num + 1) % (WANDB_LOG_FREQ_METRICS * 20) == 0:
                print(f"Step [{step_num + 1}/{NUM_ITERATIONS}], Loss: {loss_val:.4f}")

    print(f"\nTraining completed after {current_step} steps.")

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