import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from main_space.utils.settings_utils import get_dataset_config, get_hyperparameter_config, get_training_config
from main_space.data_management import load_and_process_dataset_for_lm, CausalLMTrainingDataset, InstructionFinetuningDataset, apply_chat_template
from transformers import AutoTokenizer

SPLITS = ['train', 'validation', 'test']


def get_next_batch(dataset_iter, dataloader, step_num, device):

    datasetConfig = get_dataset_config()

    new_epoch = False

    try:

        batch = next(dataset_iter)

    except StopIteration:
        new_epoch = True

        print(f"Epoch finished at step {step_num}. Resetting train_loader for continued iteration.")

        dataset_iter = iter(dataloader)
        batch = next(dataset_iter)


    input_ids = batch['input_ids'].to(device,
                                      non_blocking=True if datasetConfig.pin_memory_dataloader and device.type == "cuda" else False)
    target_ids = batch['labels'].to(device,
                                    non_blocking=True if datasetConfig.pin_memory_dataloader and device.type == "cuda" else False)

    return input_ids, target_ids, dataset_iter, new_epoch


def get_vocabulary_size(tokenizer):
    return tokenizer.get_vocab_size()


def get_dataloader(split):
    # Učitavanje svih konfiguracija na početku
    trainingConfig = get_training_config()
    datasetConfig = get_dataset_config()
    hyperParamConfig = get_hyperparameter_config()

    if split not in SPLITS:
        raise ValueError(f"Split '{split}' nije validan. Očekuje se jedan od {SPLITS}.")

    # --- GLAVNA LOGIKA GRANANJA ---
    if trainingConfig.is_finetuning:
        # ===== KOD ZA FINO PODEŠAVANJE (SLIMORCA) =====

        finetuneConfig = trainingConfig.finetuneConfig

        # 1. Učitaj AutoTokenizer i dodaj specijalne tokene za ChatML format
        loaded_tokenizer = AutoTokenizer.from_pretrained(datasetConfig.tokenizer_path)
        special_tokens_to_add = ["<|im_start|>", "<|im_end|>"]
        loaded_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

        # Postavljanje pad tokena ako ne postoji
        if loaded_tokenizer.pad_token is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

        # 2. Učitaj SlimOrca dataset s Hugging Face Huba
        # Koristimo streaming=True za brže učitavanje i manju potrošnju diska
        raw_dataset = load_dataset(finetuneConfig.finetune_dataset_name, split='train')

        # 3. Primijeni chat predložak na svaki primjer u letu
        # `map` će pozvati `apply_chat_template` za svaki redak
        formatted_dataset = raw_dataset.map(apply_chat_template, fn_kwargs={'tokenizer': loaded_tokenizer})

        # 4. Kreiraj `InstructionFinetuningDataset` koji radi maskiranje
        dataset = InstructionFinetuningDataset(
            hf_dataset=formatted_dataset,
            tokenizer=loaded_tokenizer,
            max_seq_len=hyperParamConfig.ctx_len
        )

    else:
        # ===== POSTOJEĆI KOD ZA PRE-TRENING (WIKITEXT) =====

        # Učitaj standardni tokenizer
        loaded_tokenizer = AutoTokenizer.from_file(datasetConfig.tokenizer_path)

        # TODO: provjeriti jesu li ovi tokeni ispravni za vaš BPE tokenizer
        bos_token_id = loaded_tokenizer.cls_token_id
        eos_token_id = loaded_tokenizer.sep_token_id
        pad_token_id = loaded_tokenizer.pad_token_id

        if None in [bos_token_id, eos_token_id, pad_token_id]:
            raise ValueError("Tokenizer mora imati definirane CLS, SEP i PAD tokene.")

        lm_ready_hf_datasets, _ = load_and_process_dataset_for_lm(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )

        hf_split_data = lm_ready_hf_datasets[split]
        if not hf_split_data or len(hf_split_data) == 0:
            raise ValueError(f"Preprocesirani podaci za '{split}' split su prazni.")

        dataset = CausalLMTrainingDataset(
            hf_dataset_split=hf_split_data,
            max_seq_len=hyperParamConfig.ctx_len,
            pad_token_id=pad_token_id
        )

    # --- ZAJEDNIČKI DIO: KREIRANJE DATALOADERA ---

    # Za reproducibilnost (posebno važno kod shuffle-a)
    generator = torch.Generator().manual_seed(trainingConfig.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=trainingConfig.batch_size,
        shuffle=(split == 'train'),  # Miješaj samo podatke za trening
        generator=generator,
        num_workers=datasetConfig.num_workers_dataloader,
        pin_memory=datasetConfig.pin_memory_dataloader if torch.cuda.is_available() else False
    )

    # Vraćamo i dataloader i tokenizer
    return dataloader, loaded_tokenizer
