import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from main_space.utils.settings_utils import get_dataset_config, get_hyperparameter_config, get_training_config
from main_space.data_management import load_and_process_dataset_for_lm, CausalLMTrainingDataset

SPLITS = ['train', 'validation', 'test']


def get_next_batch(train_iter, train_dataloader, step_num, device):

    datasetConfig = get_dataset_config()

    try:
        batch = next(train_iter)
    except StopIteration:
        print(f"Epoch finished at step {step_num}. Resetting train_loader for continued iteration.")
        train_iter = iter(train_dataloader)
        batch = next(train_iter)

    input_ids = batch['input_ids'].to(device,
                                      non_blocking=True if datasetConfig.pin_memory_dataloader and device.type == "cuda" else False)
    target_ids = batch['labels'].to(device,
                                    non_blocking=True if datasetConfig.pin_memory_dataloader and device.type == "cuda" else False)

    return input_ids, target_ids, train_iter


def get_vocabulary_size(tokenizer):
    return tokenizer.get_vocab_size()


def get_dataloader(split,):
    datasetConfig = get_dataset_config()
    hyperParamConfig = get_hyperparameter_config()
    trainingConfig = get_training_config()

    if split not in SPLITS:
        raise ValueError(f"In get_dataloader, split was passed as '{split}', split is expected to be one of {SPLITS}.")

    loaded_tokenizer = Tokenizer.from_file(datasetConfig.tokenizer_path)

    # TODO: fix these bos and eos tokens to proper ones
    bos_token_id = loaded_tokenizer.token_to_id("[SEP]")
    eos_token_id = loaded_tokenizer.token_to_id("[CLS]")
    pad_token_id = loaded_tokenizer.token_to_id("[PAD]")

    if None in [bos_token_id, eos_token_id, pad_token_id]:
        raise ValueError("Ensure tokenizer has <s>, </s>, and <pad> tokens defined.")

    lm_ready_hf_datasets, _ = load_and_process_dataset_for_lm(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )

    # TODO: better random seed generation overall
    generator = torch.Generator().manual_seed(trainingConfig.seed)

    hf_split_data = lm_ready_hf_datasets[split]
    if not hf_split_data or len(hf_split_data) == 0:
        raise ValueError(f"Preprocessed dataset for '{split}' is empty or not found.")

    dataset = CausalLMTrainingDataset(
        hf_dataset_split=hf_split_data,
        max_seq_len=hyperParamConfig.ctx_len,
        pad_token_id=pad_token_id
    )

    dataloader = DataLoader(
        dataset,
        batch_size=trainingConfig.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=datasetConfig.num_workers_dataloader,
        pin_memory=datasetConfig.pin_memory_dataloader if torch.cuda.is_available() else False
    )

    return dataloader
