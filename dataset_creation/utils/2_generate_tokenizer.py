import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers


def train_custom_bpe_tokenizer(
        data_files,
        vocab_size,
        output_path,
        min_frequency
):
    """
    Trains a custom Byte-Pair Encoding (BPE) tokenizer from text files.

    Args:
        data_files (list): List of paths to the text files for main_space.
        vocab_size (int): The desired vocabulary size.
        output_path (str): Path to save the trained tokenizer.
        min_frequency (int): The minimum frequency a pair should have to be merged.
    """
    print(f"Starting BPE tokenizer main_space with vocab_size={vocab_size}...")
    print(f"Training files: {data_files}")

    # 1. Initialize a Tokenizer with a BPE model
    # [UNK] is the token for out-of-vocabulary words
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Setup Normalizer
    # Standard normalization: NFKC Unicode normalization.
    # You can add others like Lowercase() or StripAccents() if needed,
    # but for LLM pre-main_space, preserving case is often preferred.
    # normalizers.Sequence allows combining multiple normalizers.
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
        # normalizers.Lowercase(), # Uncomment if you want to lowercase
        # normalizers.StripAccents() # Uncomment if you want to remove accents
    ])
    print(f"Using Normalizer: {tokenizer.normalizer}")

    # 3. Setup Pre-tokenizer
    # ByteLevel pre-tokenizer handles raw bytes, making it robust.
    # add_prefix_space=True is important for BPE to distinguish word starts
    # from characters inside words.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    print(f"Using Pre-tokenizer: {tokenizer.pre_tokenizer}")

    # 4. Setup Decoder
    # ByteLevel decoder correctly converts byte-level tokens back to readable text.
    tokenizer.decoder = decoders.ByteLevel()
    print(f"Using Decoder: {tokenizer.decoder}")

    # 5. Setup Trainer
    # BpeTrainer handles the BPE main_space process.
    # Define special tokens. These will be added to the vocabulary
    # and won't be split during tokenization.
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        # show_progress=True, # Shows a progress bar during main_space
        # initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # Provides initial characters
    )
    print(f"Using Trainer with vocab_size={vocab_size}, min_frequency={min_frequency}")
    print(f"Special tokens: {special_tokens}")

    # 6. Train the tokenizer
    print("Training the tokenizer...")
    tokenizer.train(files=data_files, trainer=trainer)
    print("Training complete.")

    # 7. Save the tokenizer
    # The tokenizer is saved as a single JSON file.
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {os.path.abspath(output_path)}")

    return tokenizer


def main(VOCAB_SIZE):
    # --- Configuration ---
    # This should be the directory where you saved the output from the download script
    data_directory = "../wikitext103_raw_corpus"

    # List of files to use for main_space. Typically, you train on the main_space set.
    # You can include validation/test if you have a very small dataset, but usually not recommended.
    train_files = [
        os.path.join(data_directory, "wikitext-103-raw-train.txt"),
        # os.path.join(data_directory, "wikitext-103-raw-validation.txt"),
        # os.path.join(data_directory, "wikitext-103-raw-test.txt")
    ]

    # Check if main_space files exist
    for f_path in train_files:
        if not os.path.exists(f_path):
            print(f"ERROR: Training file not found: {f_path}")
            print(f"Please ensure you have downloaded the WikiText-103 raw dataset to '{data_directory}'.")
            exit(1)

    VERSION = 1

    TOKENIZER_OUTPUT_PATH = ""
    # Output path for the tokenizer file
    while TOKENIZER_OUTPUT_PATH == "" or os.path.exists(TOKENIZER_OUTPUT_PATH):
        VERSION += 1
        TOKENIZER_OUTPUT_PATH = f"../tokenizer/{VERSION}_raw_wikitext103_bpe_vocab_{VOCAB_SIZE}.json"

    # --- Train ---
    trained_tokenizer = train_custom_bpe_tokenizer(
        data_files=train_files,
        vocab_size=VOCAB_SIZE,
        output_path=TOKENIZER_OUTPUT_PATH,
        min_frequency=5
    )

    # --- Test the trained tokenizer (optional) ---
    if trained_tokenizer:
        print("\n--- Testing the trained tokenizer ---")

        # Load the tokenizer from the saved file (to ensure it saved correctly)
        loaded_tokenizer = Tokenizer.from_file(TOKENIZER_OUTPUT_PATH)

        sample_text = "Hello, y'all! How are you doing today? This is a test sentence for our new BPE tokenizer."
        print(f"Original text: {sample_text}")

        # Encode the text
        encoded_output = loaded_tokenizer.encode(sample_text)

        print(f"Tokens: {encoded_output.tokens}")
        print(f"Token IDs: {encoded_output.ids}")

        # Decode the token IDs back to text
        decoded_text = loaded_tokenizer.decode(encoded_output.ids)
        print(f"Decoded text: {decoded_text}")

        # Check special tokens
        print(f"ID for [UNK]: {loaded_tokenizer.token_to_id('[UNK]')}")
        print(f"ID for [PAD]: {loaded_tokenizer.token_to_id('[PAD]')}")

        # Test unknown word
        unknown_word_text = "This text contains some very ZYXWVUTSRQPONMLKJIHGFEDCBA words."
        encoded_unknown = loaded_tokenizer.encode(unknown_word_text)
        print(f"\nOriginal text with unknown words: {unknown_word_text}")
        print(f"Tokens with unknown: {encoded_unknown.tokens}")
        print(f"Decoded unknown: {loaded_tokenizer.decode(encoded_unknown.ids)}")



if __name__ == "__main__":
    # for vs in range(1000, 20001, 1000):
    #     main(vs)

    for vs in [2000, 3000, 5000, 8000, 12000]:
        main(vs)