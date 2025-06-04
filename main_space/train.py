import torch

from main_space.train_management import get_global_train_manager
import main_space.utils.settings_utils as settings_utils
from main_space.utils.log_utils import print_model_params
from main_space.utils.checker_utils import validate_config


trainManage = get_global_train_manager()

projectConfig = settings_utils.get_project_config()
wandbConfig = settings_utils.get_wandb_config()
trainingConfig = settings_utils.get_training_config()
hyperParamConfig = settings_utils.get_training_config()


def main():

    torch.manual_seed(trainingConfig.seed)  # TODO: check for global seed configuration

    validate_config()

    trainManage.setup_device()
    trainManage.setup_wandb_run()

    with trainManage.get_profiler_context():

        trainManage.setup_train_dataloader()
        trainManage.setup_model()
        trainManage.setup_wandb_watch()

        # TODO fix this logging
        print_model_params(trainManage.model)

        trainManage.setup_optimizer()
        trainManage.setup_criterion()

        # TODO: move printing elsewhere
        print(f"\nStarting training for {trainingConfig.train_steps} iterations...")
        print(f"Train loader has ~{len(trainManage.train_dataloader)} batches of size {trainingConfig.batch_size}.")

        trainManage.init_train_iterator()
        trainManage.init_curr_step_counter()

        for step_num in range(trainingConfig.train_steps):

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
