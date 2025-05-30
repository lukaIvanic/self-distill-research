import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Assuming model.py is in the same directory or in PYTHONPATH
from model import MyTransformerLM # Ensure this import works

# --- SCRIPT SETTINGS ---
# W&B Configuration
WANDB_PROJECT_NAME = "self-distill-research" # Changed for new run
WANDB_ENTITY = "luka_newbie"  # <<< --- !!! YOU MUST CHANGE THIS !!! --- >>>
# Example: WANDB_ENTITY = "my_research_group" or your W&B username

# Model Configuration (passed to MyTransformerLM)
VOCAB_SIZE = 1000  # Dummy vocab size
D_MODEL = 128      # Embedding dimension / model dimension
NUM_HEADS = 4      # Number of attention heads
NUM_LAYERS = 2     # Number of Transformer blocks
MAX_SEQ_LEN = 64   # Max sequence length for dummy data and positional embeddings
DROPOUT_RATE = 0.1

# Training Configuration
NUM_ITERATIONS = 200000 # Increased for a bit more logging
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WANDB_LOG_FREQ_MODEL = 50 # How often to log model parameters/gradients with wandb.watch
WANDB_LOG_FREQ_METRICS = 5 # How often to log loss and other metrics

def main():
    # --- 1. W&B Initialization ---
    config = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "max_seq_len": MAX_SEQ_LEN,
        "dropout_rate": DROPOUT_RATE,
        "num_iterations": NUM_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
    wandb.init(
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY,
        config=config
    )
    print(f"W&B Run URL: {wandb.run.get_url()}")


    # --- 2. Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Model Instantiation ---
    model = MyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {num_params:,} trainable parameters.")
    wandb.summary["model_parameters"] = num_params # Log total params to W&B summary

    # --- 4. W&B Watch (for gradients and parameters) ---
    # log="all" logs gradients and parameters. log_freq is in steps.
    wandb.watch(model, log="all", log_freq=WANDB_LOG_FREQ_MODEL, log_graph=True) # log_graph can be True for first setup

    # --- 5. Optimizer and Loss Criterion Setup ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # For language modeling, CrossEntropyLoss is typical.
    # It expects logits of shape (N, C, ...) and targets of shape (N, ...) where C is number of classes.
    # Our model outputs (batch, seq_len, vocab_size), targets are (batch, seq_len)
    # We'll need to reshape for loss: (batch * seq_len, vocab_size) and (batch * seq_len)
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting dummy training for {NUM_ITERATIONS} iterations...")
    # --- 6. Training Loop ---
    for step in range(NUM_ITERATIONS):
        model.train() # Set model to training mode

        # --- 6a. Dummy Data Generation ---
        # Input tokens for the model
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN), device=device)
        # Target tokens (e.g., next token prediction)
        # For simplicity, let's make targets also random. In a real scenario, targets would be input_ids shifted.
        target_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN), device=device)

        # --- 6b. Forward Pass ---
        logits = model(input_ids) # Shape: [BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE]

        # --- 6c. Loss Calculation ---
        # Reshape for CrossEntropyLoss:
        # Logits: [BATCH_SIZE * MAX_SEQ_LEN, VOCAB_SIZE]
        # Targets: [BATCH_SIZE * MAX_SEQ_LEN]
        loss = criterion(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

        # --- 6d. Backward Pass & Optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- 6e. W&B Logging ---
        if (step + 1) % WANDB_LOG_FREQ_METRICS == 0:
            log_data = {
                "train_loss": loss.item(),
                "iteration": step + 1,
                "learning_rate": LEARNING_RATE # Fixed for now, will change with scheduler
            }
            # Log GPU memory if available
            if torch.cuda.is_available():
                log_data["gpu_mem_alloc"] = torch.cuda.memory_allocated(device) / (1024**2) # MB
                log_data["gpu_mem_cached"] = torch.cuda.memory_reserved(device) / (1024**2) # MB
            wandb.log(log_data, step=step+1)

            if (step + 1) % (WANDB_LOG_FREQ_METRICS * 10) == 0: # Print to console less frequently
                 print(f"Step [{step+1}/{NUM_ITERATIONS}], Loss: {loss.item():.4f}")


    print("Dummy training completed.")
    # --- 7. W&B Finish ---
    wandb.finish()
    print("W&B run finished.")

if __name__ == "__main__":
    # IMPORTANT: Set your WANDB_ENTITY before running!
    if WANDB_ENTITY == "your_wandb_username_or_team" or WANDB_ENTITY == "":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE SET THE 'WANDB_ENTITY' VARIABLE IN `train.py` TO YOUR      !!!")
        print("!!! W&B USERNAME OR TEAM NAME BEFORE RUNNING.                        !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()