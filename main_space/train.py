import torch
import torch.nn as nn
import torch.optim as optim

from settings import Config
from main_space.model import MyTransformerLM
from main_space.utils.log_utils import enable_wandb_watch, initialize_wandb, get_profiler_or_null_context
from main_space.utils.train_utils import make_train_step
from main_space.utils.checker_utils import validate_config
from main_space.utils.dataloader_utils import get_dataloader, get_next_batch

projectConfig = Config()
wandbConfig = projectConfig.wandbConfig

trainingConfig = projectConfig.trainingConfig
hyperParamConfig = trainingConfig.hyperParamConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(trainingConfig.seed)  # TODO: check for global seed configuration

    validate_config(projectConfig)

    wandb_run = initialize_wandb(projectConfig)

    train_dataloader = get_dataloader(
        projectConfig,
        split='train'
    )

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
        enable_wandb_watch(config=projectConfig,
                           model=model,
                           wandb_run=wandb_run)

    profiler_or_null_context = get_profiler_or_null_context(ENABLE_PROFILER=wandbConfig.is_profiler_enabled,
                                                            ENABLE_WANDB=wandbConfig.is_wandb_enabled,
                                                            wandb_run=wandb_run, device=device,
                                                            wandbConfig=wandbConfig)

    with profiler_or_null_context as prof:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model instantiated with {num_params:,} trainable parameters.")

        optimizer = optim.AdamW(model.parameters(), lr=trainingConfig.peak_lr)
        # For LM, CrossEntropyLoss ignores index -100 by default, which our DataLoader uses for label padding.
        criterion = nn.CrossEntropyLoss()

        print(f"\nStarting training for {trainingConfig.train_steps} iterations...")
        print(f"Train loader has ~{len(train_dataloader)} batches of size {trainingConfig.batch_size}.")

        train_iter = iter(train_dataloader)
        current_step = 0

        for step_num in range(trainingConfig.train_steps):
            input_ids, target_ids = get_next_batch(config=projectConfig,
                                                   train_iter=train_iter,
                                                   train_dataloader=train_dataloader,
                                                   step_num=step_num,
                                                   device=device)

            make_train_step(
                config=projectConfig,
                step_num=step_num,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                batch_input_ids=input_ids,
                batch_target_ids=target_ids,
                wandb_run_obj=wandb_run,
                profiler_obj=prof
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
