# data_preparation.py
import os
import re
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset as HFDataset  # HFDataset is the class type
from tokenizers import Tokenizer
from itertools import chain
import argparse  # For command-line options in __main__

from main_space.settings import Config
from main_space.utils.settings_utils import get_dataset_config, get_hyperparameter_config

# --- Constants (Defaults, can be overridden by args in __main__) ---
DEFAULT_TOKENIZER_PATH = os.path.join(os.getcwd(),
                                      "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")  # IMPORTANT: Update this path
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-103-raw-v1"  # Raw version has less pre-processing
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(),
                                 "dataset_creation/temp_files/cache_hf_datasets")  # Cache for HuggingFace datasets library


def print_loaded_raw_dataset_sample(raw_datasets: DatasetDict, tokenizer: Tokenizer):
    train_dataset = raw_datasets.get('train')
    print(f"data_management -> train_dataset: {train_dataset}")

    small_slice = train_dataset.select(range(100))
    print(f"data_management -> small_slice: {small_slice}")

    print("\n" + "=" * 50)
    print("  Running Tokenizer Sanity Check on 10 Examples")
    print("=" * 50 + "\n")

    for i, example in enumerate(small_slice):
        original_text = example['text']

        # Skip empty lines which are common in wikitext
        if not original_text.strip():
            print(f"--- Example {i + 1} (Skipped Empty Line) ---")
            print("-" * 40 + "\n")
            continue

        print(f"--- Example {i + 1} ---")
        print(f"Original Text:\n'{original_text}'")

        # 1. Encode the text
        # The .encode() method returns an Encoding object with details like IDs and tokens
        encoding = tokenizer.encode(original_text)
        encoded_ids = encoding.ids
        print(f"\nEncoded IDs:\n{encoded_ids}")

        # 2. Decode the list of IDs back into a string
        decoded_text = tokenizer.decode(encoded_ids)
        print(f"\nDecoded Text:\n'{decoded_text}'")

        print("\n--- Verification ---")
        if original_text == decoded_text:
            print("✅ Success: Original and decoded texts are identical.")
        else:
            print("⚠️ Warning: Original and decoded texts DO NOT match!")
            print(f"Original len: {len(original_text)}, Decoded len: {len(decoded_text)}")

        print("-" * 40 + "\n")






def print_documents_from_dataset(processed_docs, tokenizer):
    print(f"\n--- [DEBUG] Batch Processed. Found {len(processed_docs)} complete documents. ---")
    for i, doc_tokens in enumerate(processed_docs):
        # We want to decode only the content, not the special BOS/EOS tokens.
        # So we select the list of tokens from the second element to the second-to-last.
        decoded_text = tokenizer.decode(doc_tokens)

        # For readability, let's shorten very long documents.
        if len(decoded_text) > 150:
            shortened_text = f"{decoded_text[:75]} ... {decoded_text[-75:]}"
        else:
            shortened_text = decoded_text

        # Print in your requested format
        print(f"\n[Doc {i + 1}]: {shortened_text}")
    print("\n--- [DEBUG] End of Batch Report ---")

def _separate_into_docs_and_tokenize(examples, tokenizer, bos_token_id, eos_token_id):
    processed_docs = []

    curr_doc = [bos_token_id]

    for i, doc_text in enumerate(examples['text']):

        is_new_doc = None

        stripped = doc_text.strip()
        if len(stripped) > 2 and stripped[0:2] == "= " and stripped[-2:] == " =":
            is_new_doc = re.match(r" = ([^=;]+) = \n", doc_text)

            if (is_new_doc is not None) and i > 0 and (examples['text'][i-1].strip()):
                is_new_doc = False


        if is_new_doc is not None:
            curr_doc.append(eos_token_id)
            if len(curr_doc) > 30:
                processed_docs.append(curr_doc)

            curr_doc = [bos_token_id]

        if doc_text.strip():
            curr_doc += tokenizer.encode(doc_text).ids

    curr_doc.append(eos_token_id)
    processed_docs.append(curr_doc)

    #print_documents_from_dataset(processed_docs, tokenizer)

    return {"input_ids_per_doc": processed_docs}


def _group_tokenized_docs_into_blocks(examples, ctx_len, tokenizer):

    concatenated_ids = list(chain.from_iterable(examples))
    total_length = len(concatenated_ids)

    blocks = []
    for i in range(0, total_length, ctx_len):
        block = concatenated_ids[i: i + ctx_len]

        #print(f"block: {tokenizer.decode(block, skip_special_tokens=False)}")
        if block:
            if len(block) == ctx_len:
                blocks.append(block)


    return {"input_ids": blocks}

def load_and_process_dataset_for_lm(
        bos_token_id: int,
        eos_token_id: int
):
    dataset_config = get_dataset_config()
    hyperparamConfig = get_hyperparameter_config()

    if not os.path.exists(dataset_config.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {dataset_config.tokenizer_path}.")

    tokenizer = Tokenizer.from_file(dataset_config.tokenizer_path)
    raw_datasets_cache_path = os.path.join(DEFAULT_CACHE_DIR, "raw", dataset_config.dataset_name,
                                           dataset_config.dataset_config)

    raw_datasets: DatasetDict = load_dataset(dataset_config.dataset_name,
                                             dataset_config.dataset_config,
                                             cache_dir=raw_datasets_cache_path)


    #print_loaded_raw_dataset_sample(raw_datasets, tokenizer)

    tokenized_docs_cache_base = os.path.join(DEFAULT_CACHE_DIR, "tokenized_per_doc"
                                             , dataset_config.dataset_name,
                                             dataset_config.dataset_config)
    if not os.path.exists(tokenized_docs_cache_base):
        os.makedirs(tokenized_docs_cache_base)



    tokenized_dataset_docs = raw_datasets.map(
        _separate_into_docs_and_tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing individual documents",
        cache_file_names={k: os.path.join(tokenized_docs_cache_base, f"{k}.arrow") for k in raw_datasets.keys()},
        load_from_cache_file=True,
        fn_kwargs={
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "tokenizer": tokenizer,
        }
    )



    blocked_datasets_cache_base = os.path.join(DEFAULT_CACHE_DIR, "blocked_data", dataset_config.dataset_name,
                                               dataset_config.dataset_config,
                                               f"seqlen{hyperparamConfig.ctx_len}")
    if not os.path.exists(blocked_datasets_cache_base):
        os.makedirs(blocked_datasets_cache_base)

    lm_datasets = tokenized_dataset_docs.map(
        _group_tokenized_docs_into_blocks,
        batched=True,
        input_columns=["input_ids_per_doc"],
        remove_columns=["input_ids_per_doc"],
        desc=f"Grouping texts into blocks of {hyperparamConfig.ctx_len}",
        cache_file_names={k: os.path.join(blocked_datasets_cache_base, f"{k}.arrow") for k in
                          tokenized_dataset_docs.keys()},
        load_from_cache_file=True,

        fn_kwargs={
            "ctx_len": hyperparamConfig.ctx_len,
            "tokenizer": tokenizer
        }
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
