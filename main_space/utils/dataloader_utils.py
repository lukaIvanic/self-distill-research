import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from main_space.data_management import load_and_process_dataset_for_lm, CausalLMTrainingDataset

SPLITS = ['train', 'validation', 'test']

def get_next_batch(train_iter, train_dataloader, step_num, device, PIN_MEMORY_DATALOADER):
    try:
        batch = next(train_iter)
    except StopIteration:
        print(f"Epoch finished at step {step_num}. Resetting train_loader for continued iteration.")
        train_iter = iter(train_dataloader)
        batch = next(train_iter)

    input_ids = batch['input_ids'].to(device,
                                      non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)
    target_ids = batch['labels'].to(device,
                                    non_blocking=True if PIN_MEMORY_DATALOADER and device.type == "cuda" else False)

    return input_ids, target_ids

def get_vocabulary_size(tokenizer):
    return tokenizer.get_vocab_size()


def get_dataloader(tokenizer_path, ctx_len, dataset_name, dataset_config, cache_dir,
                   batch_size, seed, num_workers, pin_memory, split):
    if split not in SPLITS:
        raise ValueError(f"In get_dataloader, split was passed as '{split}', split is expected to be one of {SPLITS}.")

    loaded_tokenizer = Tokenizer.from_file(tokenizer_path)

    # TODO: fix these bos and eos tokens to proper ones
    bos_token_id = loaded_tokenizer.token_to_id("[SEP]")
    eos_token_id = loaded_tokenizer.token_to_id("[CLS]")
    pad_token_id = loaded_tokenizer.token_to_id("[PAD]")

    if None in [bos_token_id, eos_token_id, pad_token_id]:
        raise ValueError("Ensure tokenizer has <s>, </s>, and <pad> tokens defined.")

    actual_vocab_size = loaded_tokenizer.get_vocab_size()

    lm_ready_hf_datasets, _ = load_and_process_dataset_for_lm(
        tokenizer_path=tokenizer_path,
        max_seq_len=ctx_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        cache_dir=cache_dir,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )

    generator = torch.Generator().manual_seed(seed)

    hf_split_data = lm_ready_hf_datasets[split]
    if not hf_split_data or len(hf_split_data) == 0:
        raise ValueError(f"Preprocessed dataset for '{split}' is empty or not found.")

    dataset = CausalLMTrainingDataset(
        hf_dataset_split=hf_split_data,
        max_seq_len=ctx_len,
        pad_token_id=pad_token_id
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False
    )

    return dataloader
