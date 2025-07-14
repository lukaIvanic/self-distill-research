import torch
from torch.profiler import record_function

from main_space.validation_management import get_global_validation_manager, ValidationManager
import main_space.utils.settings_utils as settings_utils
from main_space.utils.log_utils import print_model_params
from main_space.utils.checker_utils import validate_config

validationManage: ValidationManager = get_global_validation_manager()

trainingConfig = settings_utils.get_training_config()
wandbConfig = settings_utils.get_wandb_config()
checkpointConfig = settings_utils.get_checkpoint_config()

def setupForValidation():
    checkpointConfig.checkpoint_frequency = float('inf')
    checkpointConfig.attempt_load_checkpoint_if_exists = True

    alias = "30M_classic:run_e39zl7c0_step_11999".split(":")
    checkpointConfig.artifact_base_name = alias[0]
    checkpointConfig.alias_to_load = alias[1]
    wandbConfig.wandb_log_freq_metrics = 5


def main():
    # TODO: move to TrainManage, and check for perfect globality
    torch.manual_seed(trainingConfig.seed)



    validate_config()

    validationManage.setup_device()
    validationManage.setup_wandb_run()

    with validationManage.get_profiler_context():

        validationManage.setup_model()
        validationManage.setup_wandb_watch()
        validationManage.setup_criterion()



        # TODO fix this logging
        print_model_params(validationManage.model)

        validationManage.setup_validation_dataloader()
        validationManage.init_validation_iterator()
        validationManage.init_curr_step_counter()

        setupForValidation()
        validationManage.attempt_load_checkpoint_if_exists()

        validationManage.setup_for_loss_calculation()

        # TODO: move printing elsewhere
        print(f"\nValidation loader has ~{len(validationManage.validation_dataloader)} batches of size {trainingConfig.batch_size}.")


        while validationManage.next_batch():
            validationManage.make_validation_step()



    # TODO: manage logging better
    print(f"\nValidation completed after {validationManage.current_validation_step} steps.")

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
