import os

import random
import numpy as np
import torch
from torch.profiler import record_function

# --- End of setup ---
from main_space.train_management import get_global_train_manager
import main_space.utils.settings_utils as settings_utils
from main_space.utils.log_utils import print_model_params, print_teacher_model_params
from main_space.utils.checker_utils import validate_config

from tokenizers import Tokenizer


def print_input_text(input_tensor: torch.Tensor):

    tokenizer_path = os.path.join(os.getcwd(), "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    """
    Prints a comprehensive set of details about a PyTorch tensor.

    Args:
        input_tensor: The PyTorch tensor to inspect.
    """
    if not isinstance(input_tensor, torch.Tensor):
        print(f"Input is not a PyTorch tensor. It is of type: {type(input_tensor)}")
        return

    # print("--- General Properties ---")
    # print(f"Type: {type(input_tensor)}")
    # print(f"Data Type (dtype): {input_tensor.dtype}")
    # print(f"Shape (size): {input_tensor.shape}")
    # print(f"Number of elements (numel): {input_tensor.numel()}")
    # print(f"Device: {input_tensor.device}")
    #
    # print("\n--- Memory and Layout ---")
    # print(f"Memory Layout: {input_tensor.layout}")
    # if input_tensor.is_contiguous():
    #     print("Is Contiguous: True")
    # else:
    #     print("Is Contiguous: False")
    #     print(f"Memory format for non-contiguous tensor might be, for example, {torch.channels_last}")
    #
    # # Storage provides a view into the underlying 1D data array.
    # if input_tensor.storage() is not None:
    #     print(f"Underlying storage type: {input_tensor.storage().type()}")
    #     print(f"Storage size: {len(input_tensor.storage())}")
    #     print(f"Storage device: {input_tensor.storage().device}")


    input_text = tokenizer.decode(list(input_tensor[0]), skip_special_tokens=False)
    print(f"input_text: {input_text[:20]} ... {input_text[-20:]}")





def main():
    trainManage = get_global_train_manager()

    trainingConfig = settings_utils.get_training_config()

    # TODO: move to TrainManage, and check for perfect globality
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(trainingConfig.seed)
    np.random.seed(trainingConfig.seed)
    torch.manual_seed(trainingConfig.seed)
    torch.cuda.manual_seed(trainingConfig.seed)
    torch.cuda.manual_seed_all(trainingConfig.seed)


    validate_config()

    trainManage.setup_device()
    trainManage.setup_wandb_run()

    with trainManage.get_profiler_context():

        trainManage.setup_model()
        trainManage.setup_teacher_model()
        trainManage.setup_wandb_watch()
        trainManage.setup_optimizer()
        trainManage.setup_criterion()


        # TODO fix this logging
        print_model_params(trainManage.model)

        if trainingConfig.distill_enabled:
            print_teacher_model_params(trainManage.teacher_model)

        trainManage.setup_train_dataloader()
        # trainManage.setup_validation_dataloader()
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

            # if (step_num + 1) % 1000 == 0 or step_num == 0:
            #     print_input_text(trainManage.input_ids)

            with record_function("make_train_step"):
                trainManage.make_train_step()






    print(f"\nTraining completed after {trainManage.current_train_step} steps.")


if __name__ == "__main__":
    main()
