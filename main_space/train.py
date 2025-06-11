import torch
from torch.profiler import record_function

# --- End of setup ---
from main_space.train_management import get_global_train_manager
import main_space.utils.settings_utils as settings_utils
from main_space.utils.log_utils import print_model_params
from main_space.utils.checker_utils import validate_config

trainManage = get_global_train_manager()

trainingConfig = settings_utils.get_training_config()


def main():
    # TODO: move to TrainManage, and check for perfect globality
    torch.manual_seed(trainingConfig.seed)

    validate_config()

    trainManage.setup_device()
    trainManage.setup_wandb_run()

    with trainManage.get_profiler_context():

        trainManage.setup_model()
        trainManage.setup_wandb_watch()
        trainManage.setup_optimizer()
        trainManage.setup_criterion()



        # TODO fix this logging
        print_model_params(trainManage.model)

        trainManage.setup_train_dataloader()
        trainManage.init_train_iterator()
        trainManage.init_curr_step_counter()

        trainManage.attempt_load_checkpoint_if_exists()

        # TODO: move printing elsewhere
        print(f"\nStarting training from step {get_global_train_manager().current_train_step} until step {trainingConfig.train_steps}, for {trainingConfig.train_steps - get_global_train_manager().current_train_step} more iterations...")
        print(f"Train loader has ~{len(trainManage.train_dataloader)} batches of size {trainingConfig.batch_size}.")


        if trainManage.current_train_step >= trainingConfig.experimental_steps:
            # TODO: fix and check experimental steps
            raise ValueError("TODO: fix and check experimental steps")

        if trainingConfig.experimental_stop:
            steps = trainingConfig.experimental_steps
        else:
            steps = trainingConfig.train_steps

        for step_num in range(trainManage.current_train_step, steps):
            with record_function("getting_next_batch"):
                trainManage.next_batch()
            trainManage.make_train_step()



    # TODO: manage logging better
    print(f"\nTraining completed after {trainManage.current_train_step} steps.")

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
