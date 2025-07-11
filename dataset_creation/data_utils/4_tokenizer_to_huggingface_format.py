import os
import json
from tokenizers import Tokenizer  # To inspect the original for special tokens
from transformers import PreTrainedTokenizerFast


def convert_tokenizer_to_hf_hub_format(
        input_tokenizer_json_path,
        output_hf_directory_path,
        model_max_length=512  # A common default, adjust if needed
):
    """
    Converts a single tokenizer.json file (from Hugging Face 'tokenizers' library)
    into the full Hugging Face Hub directory structure required by 'transformers'.

    Args:
        input_tokenizer_json_path (str): Path to your trained tokenizer.json file.
        output_hf_directory_path (str): Path to the directory where the HF-formatted
                                        tokenizer will be saved.
        model_max_length (int): The maximum sequence length this tokenizer will be
                                used with. This gets saved in tokenizer_config.json.
    """
    print(f"Starting conversion of '{input_tokenizer_json_path}' to Hugging Face Hub format.")

    if not os.path.exists(input_tokenizer_json_path):
        print(f"ERROR: Input tokenizer file not found: {input_tokenizer_json_path}")
        return

    # 1. Load the original tokenizer to inspect its configuration, especially special tokens
    # This helps ensure we pass the correct special tokens to PreTrainedTokenizerFast
    try:
        original_tokenizer = Tokenizer.from_file(input_tokenizer_json_path)
    except Exception as e:
        print(f"Error loading original tokenizer from {input_tokenizer_json_path}: {e}")
        return

    # Extract special tokens that were defined during its main_space
    # The original BPE main_space script defined: ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    # We need to map them to the roles PreTrainedTokenizerFast expects.

    # Default common special tokens - adjust if your tokenizer used different strings
    # for these roles or didn't define some of them.
    unk_token = "[UNK]"
    pad_token = "[PAD]"  # Often used for padding sequences to the same length
    cls_token = "[CLS]"  # Often used for classification tasks at the beginning of a sequence
    sep_token = "[SEP]"  # Often used to separate segments
    mask_token = "[MASK]"  # Often used for masked language modeling

    # Verify if these tokens exist in the loaded tokenizer's vocab,
    # though PreTrainedTokenizerFast will add them if they are specified
    # and not already present (but it's better if they were part of original main_space).
    vocab = original_tokenizer.get_vocab()
    if unk_token not in vocab:
        print(f"Warning: unk_token '{unk_token}' not found in original vocab. It will be added.")
    if pad_token not in vocab:
        print(f"Warning: pad_token '{pad_token}' not found in original vocab. It will be added.")
    # Add similar checks for cls, sep, mask if you are strict about it

    print(f"Using special tokens for PreTrainedTokenizerFast:")
    print(f"  UNK: {unk_token}")
    print(f"  PAD: {pad_token}")
    print(f"  CLS: {cls_token}")
    print(f"  SEP: {sep_token}")
    print(f"  MASK: {mask_token}")

    # 2. Instantiate PreTrainedTokenizerFast
    # We pass the path to the .json file directly.
    # We also explicitly define the special tokens.
    try:
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=input_tokenizer_json_path,
            model_max_length=model_max_length,
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            # You can add bos_token="[BOS]", eos_token="[EOS]" if your model/tokenizer uses them
        )
        print(f"Successfully loaded into PreTrainedTokenizerFast.")
    except Exception as e:
        print(f"Error instantiating PreTrainedTokenizerFast: {e}")
        return

    # 3. Create the output directory if it doesn't exist
    os.makedirs(output_hf_directory_path, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_hf_directory_path)}")

    # 4. Save the tokenizer in the Hugging Face Hub format
    try:
        hf_tokenizer.save_pretrained(output_hf_directory_path)
        print(f"Tokenizer successfully saved in Hugging Face Hub format to '{output_hf_directory_path}'.")
    except Exception as e:
        print(f"Error saving tokenizer with save_pretrained: {e}")
        return

    print("\nThe following files should now be in the output directory:")
    print("- tokenizer.json (the core tokenizer data)")
    print("- vocab.json (vocabulary mapping tokens to IDs)")
    print("- merges.txt (for BPE/WordPiece, shows merge rules)")
    print("- special_tokens_map.json (maps roles like 'unk_token' to actual token strings)")
    print("- tokenizer_config.json (configuration like max_length, special token roles, etc.)")


def main(VOCAB_SIZE):

    VERSION = 1
    # --- Configuration ---
    # Path to your single .json tokenizer file
    INPUT_TOKENIZER_JSON = f"../tokenizer/{VERSION}_raw_wikitext103_bpe_vocab_{VOCAB_SIZE}.json"

    # Desired output directory for the Hugging Face Hub formatted tokenizer
    OUTPUT_HF_TOKENIZER_DIR = f"../tokenizer/bpe_hug_pub_{VOCAB_SIZE}_v{VERSION}"

    # Optional: The max sequence length your model is designed for.
    # This will be saved in tokenizer_config.json
    MODEL_MAX_LENGTH = 512

    # --- Run Conversion ---
    if not os.path.exists(INPUT_TOKENIZER_JSON):
        print(f"ERROR: The input tokenizer file '{INPUT_TOKENIZER_JSON}' was not found.")
        print("Please make sure you have run the tokenizer main_space script first and that the file exists.")
    else:
        convert_tokenizer_to_hf_hub_format(
            input_tokenizer_json_path=INPUT_TOKENIZER_JSON,
            output_hf_directory_path=OUTPUT_HF_TOKENIZER_DIR,
            model_max_length=MODEL_MAX_LENGTH
        )

        # You can test loading it back (optional)
        print("\n--- Optional: Testing loading the saved HF Hub format tokenizer ---")
        try:
            reloaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(OUTPUT_HF_TOKENIZER_DIR)
            test_sentence = "This is a test sentence."
            encoded = reloaded_tokenizer.encode(test_sentence)
            decoded = reloaded_tokenizer.decode(encoded)
            print(f"Original: {test_sentence}")
            print(f"Encoded IDs: {encoded}")
            print(f"Decoded: {decoded}")
            print(f"PAD token: {reloaded_tokenizer.pad_token}, ID: {reloaded_tokenizer.pad_token_id}")
            print("Successfully reloaded and tested the tokenizer from the HF Hub format directory.")
        except Exception as e:
            print(f"Error reloading or testing tokenizer from HF Hub directory: {e}")


if __name__ == "__main__":
    VOCAB_SIZE = 5000
    main(VOCAB_SIZE)
