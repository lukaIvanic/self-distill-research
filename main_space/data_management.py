# data_preparation.py
import os
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset as HFDataset  # HFDataset is the class type
from tokenizers import Tokenizer
from itertools import chain
import argparse  # For command-line options in __main__

from main_space.settings import Config

# --- Constants (Defaults, can be overridden by args in __main__) ---
DEFAULT_TOKENIZER_PATH = os.path.join(os.getcwd(), "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")  # IMPORTANT: Update this path
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-103-raw-v1" # Raw version has less pre-processing
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), "dataset_creation/temp_files/cache_hf_datasets")  # Cache for HuggingFace datasets library


def load_and_process_dataset_for_lm(
        config: Config,
        bos_token_id: int,
        eos_token_id: int
):

    dataset_config = config.datasetConfig
    hyperparamConfig = config.trainingConfig.hyperParamConfig

    if not os.path.exists(dataset_config.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {dataset_config.tokenizer_path}.")


    tokenizer = Tokenizer.from_file(dataset_config.tokenizer_path)
    raw_datasets_cache_path = os.path.join(DEFAULT_CACHE_DIR, "raw", dataset_config.dataset_name, dataset_config.dataset_config)
    raw_datasets = load_dataset(dataset_config.dataset_name, dataset_config.dataset_config, cache_dir=raw_datasets_cache_path)
    def tokenize_individual_docs(examples):
        processed_docs = []
        for doc_text in examples['text']:
            if not doc_text.strip():
                continue
            token_ids = [bos_token_id]
            token_ids.extend(tokenizer.encode(doc_text).ids)
            token_ids.append(int(eos_token_id))
            processed_docs.append(token_ids)
        return {"input_ids_per_doc": processed_docs}

    tokenized_docs_cache_base = os.path.join(DEFAULT_CACHE_DIR, "tokenized_per_doc", dataset_config.dataset_name, dataset_config.dataset_config)
    if not os.path.exists(tokenized_docs_cache_base): 
      os.makedirs(tokenized_docs_cache_base)

    tokenized_datasets = raw_datasets.map(
        tokenize_individual_docs,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing individual documents",
        cache_file_names={k: os.path.join(tokenized_docs_cache_base, f"{k}.arrow") for k in raw_datasets.keys()},
        load_from_cache_file=True
    )

    def group_tokenized_docs_into_blocks(examples):
        # examples['input_ids_per_doc'] is a list of lists (each inner list is a tokenized doc)
        concatenated_ids = list(chain.from_iterable(examples))
        total_length = len(concatenated_ids)

        if total_length == 0:
            return {"input_ids": []}

        # Drop the remainder to ensure full blocks, or handle it.
        # Here, we process in chunks of block_size.
        # The last partial block will be included.
        # The CustomTorchLMDataset will handle padding for this last block if it's shorter.
        blocks = []
        for i in range(0, total_length, hyperparamConfig.ctx_len):
            block = concatenated_ids[i: i + hyperparamConfig.ctx_len]
            if block:  # Ensure block is not empty
                blocks.append(block)

        return {"input_ids": blocks}

    blocked_datasets_cache_base = os.path.join(DEFAULT_CACHE_DIR, "blocked_data", dataset_config.dataset_name, dataset_config.dataset_config,
                                               f"seqlen{hyperparamConfig.ctx_len}")
    if not os.path.exists(blocked_datasets_cache_base): os.makedirs(blocked_datasets_cache_base)
    print(blocked_datasets_cache_base)
    # For this map to be most effective for caching the final blocks,
    # it should ideally operate on the entire dataset split at once if memory allows,
    # or process it in a way that the cache represents the final blocked structure.
    # By mapping over `tokenized_datasets` (which is already loaded from cache or computed),
    # this step transforms the list of tokenized docs into a list of blocks.
    # The `batched=True` along with how `datasets` handles large inputs to `map`
    # (often sending the whole thing if no `batch_size` is specified in map)
    # makes this efficient. The result of this `.map` is then cached.

    lm_datasets = tokenized_datasets.map(
        group_tokenized_docs_into_blocks,
        batched=True,  # Process all documents in a split together for this map
        # batch_size can be set to a large number or -1 (though -1 is not an official API for batch_size)
        # to encourage processing the whole split if it fits. By default, batched=True uses a batch_size of 1000.
        # For creating blocks, processing the entire split ensures all docs are concatenated before blocking.
        # This is memory-intensive for `concatenated_ids` if the split is huge.
        # Given Wikitext-103's size, this should be acceptable.
        input_columns=["input_ids_per_doc"],  # Specify input column
        remove_columns=["input_ids_per_doc"],  # Remove the intermediate column
        desc=f"Grouping texts into blocks of {hyperparamConfig.ctx_len}",
        cache_file_names={k: os.path.join(blocked_datasets_cache_base, f"{k}.arrow") for k in
                          tokenized_datasets.keys()},
        load_from_cache_file=True
    )

    for split_name, ds in lm_datasets.items():
        print(f"Split '{split_name}' processed into {len(ds)} blocks.")
        if len(ds) > 0:
            print(f"  Lengths - First block: {len(ds[0]['input_ids'])}, Last block: {len(ds[-1]['input_ids'])}")

    return lm_datasets, tokenizer

class CausalLMTrainingDataset(TorchDataset):
    def __init__(self, hf_dataset_split: HFDataset, max_seq_len: int, pad_token_id: int):
        self.hf_dataset = hf_dataset_split
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        token_ids = self.hf_dataset[idx]['input_ids']  # This is a list of ints

        # Input sequence (first max_seq_len tokens)
        input_ids_chunk = token_ids[:self.max_seq_len]

        # Label sequence (shifted by one, also up to max_seq_len tokens)
        labels_chunk = token_ids[1:self.max_seq_len + 1]

        # Pad input_ids if shorter than max_seq_len
        input_padding_needed = self.max_seq_len - len(input_ids_chunk)
        if input_padding_needed > 0:
            input_ids_chunk.extend([self.pad_token_id] * input_padding_needed)

        # Pad labels if shorter than max_seq_len, using -100 for ignore_index
        label_padding_needed = self.max_seq_len - len(labels_chunk)
        if label_padding_needed > 0:
            labels_chunk.extend([-100] * label_padding_needed)

        return {
            "input_ids": torch.tensor(input_ids_chunk, dtype=torch.long),
            "labels": torch.tensor(labels_chunk, dtype=torch.long)
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare and test Causal LM dataset.")
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH,
                        help="Path to the trained tokenizer file.")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN, help="Maximum sequence length.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                        help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG, help="Configuration of the dataset.")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Directory for Hugging Face datasets library to cache raw and processed data.")
    parser.add_argument("--view_n_batches", type=int, default=2,
                        help="Number of batches to view per epoch for testing.")
    parser.add_argument("--view_n_samples_in_batch", type=int, default=1,
                        help="Number of samples to decode and print from a viewed batch.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Set seed for reproducibility of shuffling in DataLoader
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)  # For DataLoader

    print(f"Using tokenizer: {args.tokenizer_path}")
    # Load tokenizer once to get special token IDs
    if not os.path.exists(args.tokenizer_path):
        print(f"ERROR: Tokenizer not found at {args.tokenizer_path}. Please provide correct path.")
        exit()

    temp_tokenizer = Tokenizer.from_file(args.tokenizer_path)
    # bos_token_id = temp_tokenizer.token_to_id("<s>")
    bos_token_id = temp_tokenizer.token_to_id("[SEP]")
    eos_token_id = temp_tokenizer.token_to_id("[CLS]")
    pad_token_id = temp_tokenizer.token_to_id("[PAD]")

    if None in [bos_token_id, eos_token_id, pad_token_id]:
        print("Error: One or more special tokens (<s>, </s>, <pad>) not found in the tokenizer.")
        print(f"  Found: <s> id: {bos_token_id}, </s> id: {eos_token_id}, <pad> id: {pad_token_id}")
        exit()

    print(f"Special token IDs - BOS (<s>): {bos_token_id}, EOS (</s>): {eos_token_id}, PAD (<pad>): {pad_token_id}")

    # 1. Load and process Hugging Face Datasets (uses caching)
    # This returns a DatasetDict where each split is a HFDataset of blocks
    lm_ready_hf_datasets, loaded_tokenizer = load_and_process_dataset_for_lm(
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )

    # 2. Create PyTorch Datasets and DataLoaders
    pytorch_datasets = {}
    dataloaders = {}

    for split_name, hf_split_data in lm_ready_hf_datasets.items():
        if not hf_split_data:  # Skip if a split is empty after processing
            print(f"Split '{split_name}' is empty or not found, skipping DataLoader creation.")
            continue
        pytorch_datasets[split_name] = CausalLMTrainingDataset(
            hf_dataset_split=hf_split_data,
            max_seq_len=args.max_seq_len,
            pad_token_id=pad_token_id
        )
        dataloaders[split_name] = DataLoader(
            pytorch_datasets[split_name],
            batch_size=args.batch_size,
            shuffle=(split_name == 'train'),  # Only shuffle the training set
            generator=generator if split_name == 'train' else None  # Use generator for reproducible shuffle
        )

    print("\n--- Dataset and DataLoader Test ---")
    if 'train' not in dataloaders:
        print("No training DataLoader created. Exiting test.")
        exit()

    first_batch_epoch1 = None
    first_batch_epoch2 = None

    for epoch in range(2):  # Simulate 2 epochs
        print(f"\n--- Epoch {epoch + 1} ---")
        # Important: For shuffle to change between epochs, DataLoader needs to be re-iterated
        # or re-created if its internal sampler state isn't reset.
        # With a fixed generator, the shuffle order will be the same for each "true" new iteration
        # over the DataLoader if it's the same DataLoader instance.
        # To get different shuffles per epoch with a fixed seed for the whole run, you'd
        # typically re-seed a generator or use a new one for each epoch's DataLoader, or
        # rely on the fact that if you iterate fully, the next epoch starts fresh.
        # For this test, the `generator` object is passed, so shuffle order is fixed per "full pass".

        # To demonstrate different batches if shuffle=True, we just iterate.
        # If we were to save sampler state and resume, that's more complex.
        # The current setup with DataLoader(shuffle=True, generator=generator) will produce
        # the same shuffled order each time this script is run, for a given epoch.
        # If you want a different shuffle for epoch 2 *within the same run*,
        # you'd typically just let the DataLoader iterate.

        for i, batch in enumerate(dataloaders['train']):
            if i >= args.view_n_batches:
                break

            print(f"Epoch {epoch + 1}, Batch {i + 1}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")  # Should be [batch_size, max_seq_len]
            print(f"  labels shape:    {batch['labels'].shape}")

            if epoch == 0 and i == 0:
                first_batch_epoch1 = batch['input_ids'].clone()
            if epoch == 1 and i == 0:
                first_batch_epoch2 = batch['input_ids'].clone()

            if i < args.view_n_batches:  # Only decode for the first few batches requested
                for sample_idx in range(min(args.view_n_samples_in_batch, args.batch_size)):
                    print(f"  --- Sample {sample_idx + 1} from Batch {i + 1} (Epoch {epoch + 1}) ---")
                    input_tokens = batch['input_ids'][sample_idx].tolist()
                    label_tokens = batch['labels'][sample_idx].tolist()

                    # Filter out padding for decoding display if desired, but show some for context
                    # For input_ids, pad_token_id is used. For labels, -100 is used.

                    print(f"    Input Tokens (first 50): {input_tokens[:50]}...")
                    decoded_input = loaded_tokenizer.decode([t for t in input_tokens if t != pad_token_id])
                    print(f"    Decoded Input (no pad): '{decoded_input[:200]}...'")

                    # Show how padding looks in labels
                    # Find first occurrence of -100 to show context around padding
                    try:
                        first_pad_label_idx = label_tokens.index(-100)
                        print(
                            f"    Label Tokens (around padding): ...{label_tokens[max(0, first_pad_label_idx - 20):first_pad_label_idx + 20]}...")
                    except ValueError:  # No -100 found
                        print(f"    Label Tokens (first 50): {label_tokens[:50]}...")

                    # Check how many actual content tokens vs padding in this sample
                    num_pad_inputs = input_tokens.count(pad_token_id)
                    num_pad_labels = label_tokens.count(-100)
                    print(f"    Input padding count: {num_pad_inputs}/{args.max_seq_len}")
                    print(f"    Label padding count: {num_pad_labels}/{args.max_seq_len}")

    # Compare first batches of different epochs (if shuffle=True and iterations happened)
    if first_batch_epoch1 is not None and first_batch_epoch2 is not None:
        if dataloaders['train']:  # Only makes sense if shuffle is True
            are_first_batches_different = not torch.equal(first_batch_epoch1, first_batch_epoch2)
            print(f"\nFirst batch of epoch 1 and epoch 2 are different (due to shuffle): {are_first_batches_different}")
            if not are_first_batches_different:
                print(
                    "  Note: If shuffle=True, they should be different. If they are the same, ensure DataLoader is fully iterated or re-seeded correctly for 'true' epochs in a long run.")
        else:
            print("\nShuffle was not enabled for training DataLoader, so first batches across epochs will be the same.")

    print("\n--- Validation Set Quick Check (first batch) ---")
    if 'validation' in dataloaders and len(dataloaders['validation']) > 0:
        val_batch = next(iter(dataloaders['validation']))
        print(f"Validation Batch 1 input_ids shape: {val_batch['input_ids'].shape}")
        print(f"Validation Batch 1 labels    shape: {val_batch['labels'].shape}")
        input_tokens_val = val_batch['input_ids'][0].tolist()
        print(
            f"  Decoded Val Input Sample (no pad): '{loaded_tokenizer.decode([t for t in input_tokens_val if t != pad_token_id])[:200]}...'")

    else:
        print("No validation data/loader to check.")

    print("\nData preparation script test finished.")