import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.profiler import profile, ProfilerActivity

import contextlib

from settings import Config
from main_space.model import MyTransformerLM
from main_space.utils.log_utils import enable_wandb_watch, initialize_wandb, get_profiler_or_null_context
from main_space.utils.train_utils import calculate_lr, make_train_step, validation_run
from main_space.utils.checker_utils import validate_config
from main_space.utils.dataloader_utils import get_dataloader, get_next_batch



projectConfig = Config()
wandbConfig = projectConfig.wandbConfig

trainingConfig = projectConfig.trainingConfig
hyperParamConfig = trainingConfig.hyperParamConfig

STRICT_ERROR = True
#
# # Model Configuration (passed to MyTransformerLM)
# VOCAB_SIZE = 5000  # Dummy vocab size
# D_MODEL = 1  # Embedding dimension / model dimension
# NUM_HEADS = 1  # Number of attention heads
# NUM_LAYERS = 1  # Number of Transformer blocks
# CTX_LEN = 32  # Max sequence length for dummy data and positional embeddings
# DROPOUT_RATE = 0.1
#
# """
# Testing for 3k steps:
# lr | batch_size
# 1e-3 | 256
# 1e-3 | 512
# 1e-3 | 1024
# 3e-4 | 256
# 3e-4 | 512
# 3e-4 | 1024
# 1e-4 | 256
# 1e-4 | 512
# 1e-4 | 1024
# """
#
# # Training Configuration
# WARMUP_STEPS = int(1e2)
# TRAIN_STEPS = int(1e3)
# LEARNING_RATE = 1e-3
# BATCH_SIZE = 1
# SCHEDULER_TYPE = "cosine"  # Options: "cosine", "inverse_sqrt", "linear"
# MIN_LEARNING_RATE = 1e-5
# SEED = 42

# --- Precision Configuration ---
PRECISION = "float32"  # Options: "float32", "float16", "bfloat16"
# "float16" uses GradScaler.
# "bfloat16" generally doesn't require GradScaler but can be used.
# Performance and support for bfloat16 depend on the GPU.

# --- Data and Tokenizer Configuration ---
TOKENIZER_PATH = os.path.join(os.getcwd(),
                              "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")  # IMPORTANT: UPDATE THIS
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
CACHE_DIR = os.path.join(os.getcwd(), "/dataset_creation/temp_files/cache_hf_datasets")
# DataLoader options
NUM_WORKERS_DATALOADER = 0  # 0 for main process, >0 for multiprocessing. Start with 0 for simplicity/debugging.
PIN_MEMORY_DATALOADER = True
VOCAB_SIZE_FROM_TOKENIZER = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(trainingConfig.seed)  # TODO: check for global seed configuration

    validate_config(device=device,
                    ENABLE_WANDB=wandbConfig.is_wandb_enabled,
                    ENABLE_PROFILER=wandbConfig.is_profiler_enabled,
                    WANDB_WATCH_LEVEL=wandbConfig.wandb_watch_level,
                    TOKENIZER_PATH=TOKENIZER_PATH,
                    PRECISION=PRECISION,
                    STRICT_ERROR=STRICT_ERROR)

    if PRECISION == "float16":
        precision_dtype = torch.float16
        scaler = GradScaler(device='cuda', enabled=True)
        print(f"Using Automatic Mixed Precision with dtype: {precision_dtype} and GradScaler.")
    elif PRECISION == "bfloat16":
        precision_dtype = torch.bfloat16
        scaler = None
        print(f"Using Automatic Mixed Precision with dtype: {precision_dtype}.")
    elif PRECISION == "float32":
        precision_dtype = torch.float32
        scaler = None
        print("Using float32 precision.")

    wandb_run = initialize_wandb(WANDB_PROJECT_NAME=wandbConfig.wandb_project_name, WANDB_ENTITY=wandbConfig.wandb_entity,
                                 trainingConfig=trainingConfig, DATASET_NAME=DATASET_NAME, DATASET_CONFIG=DATASET_CONFIG,
                                 ENABLE_PROFILER=wandbConfig.is_profiler_enabled, PRECISION=PRECISION,
                                 precision_dtype=precision_dtype,
                                 wandbConfig=wandbConfig)

    train_dataloader = get_dataloader(
        tokenizer_path=TOKENIZER_PATH,
        ctx_len=hyperParamConfig.ctx_len,
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        cache_dir=CACHE_DIR,
        batch_size=trainingConfig.batch_size,
        seed=trainingConfig.seed,
        num_workers=NUM_WORKERS_DATALOADER,
        pin_memory=PIN_MEMORY_DATALOADER,
        split='train'
    )

    # --- 3. Model Instantiation ---
    model = MyTransformerLM(
        vocab_size=hyperParamConfig.vocab_size,  # TODO: use vocab size directly from tokenizer or ensure it's correct
        d_model=hyperParamConfig.d_model,
        n_heads=hyperParamConfig.num_heads,
        n_layers=hyperParamConfig.num_layers,
        ctx_size=hyperParamConfig.ctx_len,
        p_dropout=hyperParamConfig.dropout_rate
    ).to(device)

    # --- 4. W&B Watch (if enabled and level is not "none") ---
    if wandbConfig.is_wandb_enabled and wandb_run and wandbConfig.wandb_watch_level != "none":
        enable_wandb_watch(model=model,
                           ENABLE_WANDB=wandbConfig.is_wandb_enabled,
                           wandb_run=wandb_run,
                           WANDB_WATCH_LEVEL=wandbConfig.wandb_watch_level,
                           WANDB_LOG_FREQ_MODEL_WATCH=wandbConfig.wandb_log_freq_model_watch,
                           WANDB_LOG_GRAPH=wandbConfig.does_wandb_log_graph)

    profiler_or_null_context = get_profiler_or_null_context(ENABLE_PROFILER=wandbConfig.is_profiler_enabled, ENABLE_WANDB=wandbConfig.is_wandb_enabled,
                                                            wandb_run=wandb_run, device=device,
                                                            wandbConfig=wandbConfig)

    with profiler_or_null_context as prof:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model instantiated with {num_params:,} trainable parameters.")

        optimizer = optim.AdamW(model.parameters(), lr=trainingConfig.learning_rate)
        # For LM, CrossEntropyLoss ignores index -100 by default, which our DataLoader uses for label padding.
        criterion = nn.CrossEntropyLoss()

        print(f"\nStarting training for {trainingConfig.train_steps} iterations...")
        print(f"Train loader has ~{len(train_dataloader)} batches of size {trainingConfig.batch_size}.")

        train_iter = iter(train_dataloader)
        current_step = 0

        for step_num in range(trainingConfig.train_steps):

            input_ids, target_ids = get_next_batch(train_iter=train_iter,
                                                   train_dataloader=train_dataloader,
                                                   step_num=step_num,
                                                   device=device,
                                                   PIN_MEMORY_DATALOADER=PIN_MEMORY_DATALOADER)


            make_train_step(
                step_num, model, criterion, optimizer, device,
                input_ids, target_ids,
                wandbConfig.is_wandb_enabled, wandb_run, profiler_obj=prof,
                torch_dtype=precision_dtype,
                scaler=scaler,
                PRECISION=PRECISION,
                wandbConfig=wandbConfig,
                trainingConfig=trainingConfig
            )
            current_step += 1

    print(f"\nTraining completed after {current_step} steps.")

    # TODO figure out validation run
    # validation_run(model=model,
    #                val_loader=val_loader,
    #                criterion=criterion,
    #                device=device,
    #                wandb=wandb,
    #                wandb_run=wandb_run,
    #                ENABLE_WANDB=ENABLE_WANDB,
    #                PIN_MEMORY_DATALOADER=PIN_MEMORY_DATALOADER,)


if __name__ == "__main__":
    main()
