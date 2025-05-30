import os
import random
from tokenizers import Tokenizer


def inspect_tokenizer_output_detailed(
        tokenizer_path,
        training_data_file,
        num_samples,
        max_save_lines,
        original_output_file,
        piped_tokens_output_file,
        decoded_output_file  # New file for fully decoded text
):
    """
    Loads a tokenizer, samples lines from main_space data, tokenizes them,
    joins tokens with '|', decodes them, and saves original, piped, and decoded versions.

    Args:
        tokenizer_path (str): Path to the trained tokenizer .json file.
        training_data_file (str): Path to the raw main_space text file.
        num_samples (int): Number of random lines to sample and process.
        original_output_file (str): File to save the original sampled lines.
        piped_tokens_output_file (str): File to save the tokenized lines with '|' separator.
        decoded_output_file (str): File to save the fully decoded (natural language) lines.
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer file not found at {tokenizer_path}")
        return
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Reading main_space data from: {training_data_file}")
    if not os.path.exists(training_data_file):
        print(f"ERROR: Training data file not found at {training_data_file}")
        return

    try:
        with open(training_data_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"Error reading main_space data file: {e}")
        return

    if not all_lines:
        print("Training data file is empty.")
        return

    num_samples = min(num_samples, len(all_lines))
    if num_samples == 0:
        print("No lines to sample.")
        return

    sampled_lines = random.sample(all_lines, num_samples)

    original_texts_to_save = []
    piped_tokens_to_save = []
    decoded_texts_to_save = []

    totalWords = 0
    totalTokens = 0

    print(f"\n--- Processing {num_samples} sampled lines ---")
    for i, line in enumerate(sampled_lines):
        original_line = line.strip()
        if not original_line:
            continue

        original_line = " " + original_line


        original_texts_to_save.append(original_line)

        # Tokenize
        encoded = tokenizer.encode(original_line)
        tokens = encoded.tokens  # e.g., ['ĠMor', 'an', 'Ġhas', ...]

        totalWords += len(original_line.split(" "))
        totalTokens += len(tokens)


        # Create piped version (shows raw tokens with Ġ)
        piped_version = "|".join(tokens)
        piped_version = piped_version.replace('Ġ', ' ')
        piped_tokens_to_save.append(piped_version)




        # Detokenize using the tokenizer's decode method
        # This is the crucial step for getting natural-looking text with correct spaces
        detokenized_line = tokenizer.decode(encoded.ids)
        decoded_texts_to_save.append(detokenized_line)

        if i < 0:  # Print a few examples to console for immediate feedback
            print(f"\nSample {i + 1}:")
            print(f"  Original    : {original_line}")
            print(f"  Tokens      : {tokens}")
            print(f"  Piped Tokens: {piped_version}")  # This will have Ġ
            print(f"  Decoded Text: {detokenized_line}")  # This should look like normal language

    # Save to files, ensuring UTF-8 encoding
    try:
        with open(original_output_file, "w", encoding="utf-8") as f:
            for i, text in enumerate(original_texts_to_save):
                if i == max_save_lines:
                    break
                f.write(text + "\n")
        print(f"\nSaved original sampled text to: {os.path.abspath(original_output_file)}")

        with open(piped_tokens_output_file, "w", encoding="utf-8") as f:
            for i, text in enumerate(piped_tokens_to_save):
                if i == max_save_lines:
                    break
                f.write(text + "\n")
        print(f"Saved piped tokenized text ('Ġ' replaced with ' ') to: {os.path.abspath(piped_tokens_output_file)}")

        with open(decoded_output_file, "w", encoding="utf-8") as f:
            for i, text in enumerate(decoded_texts_to_save):
                if i == max_save_lines:
                    break
                f.write(text + "\n")
        print(f"Saved fully decoded text (normal spaces) to: {os.path.abspath(decoded_output_file)}")

    except Exception as e:
        print(f"Error saving output files: {e}")


    print(f"Total words in sampled text: {totalWords}")
    print(f"Total tokens in sampled text: {totalTokens}")
    if totalWords > 0:
        print(f"Token to word ratio (lower is better): {totalTokens/totalWords:.3f}")
    else:
        print(f"Can't calculate token to word ratio because there are {totalWords} total words.")

def main(VOCAB_SIZE):
    # --- Configuration ---
    VERSION = 4
    TOKENIZER_FILE_PATH = f"../tokenizer/{VERSION}_raw_wikitext103_bpe_vocab_{VOCAB_SIZE}.json"
    TRAINING_DATA_FILE = os.path.join("../wikitext103_raw_corpus", "wikitext-103-raw-train.txt")
    # TRAINING_DATA_FILE = os.path.join("../wikitext103_raw_corpus", "wikitext-103-raw-validation.txt")
    # TRAINING_DATA_FILE = os.path.join("../wikitext103_raw_corpus", "wikitext-103-raw-test.txt")
    NUMBER_OF_SAMPLES = 20000
    MAX_SAVE_LINES = 100

    # --- Run inspection ---
    inspect_tokenizer_output_detailed(
        tokenizer_path=TOKENIZER_FILE_PATH,
        training_data_file=TRAINING_DATA_FILE,
        num_samples=NUMBER_OF_SAMPLES,
        max_save_lines=MAX_SAVE_LINES,
        original_output_file="../temp_files/1_sampled_original_text.txt",
        piped_tokens_output_file="../temp_files/2_sampled_piped_tokens.txt",
        decoded_output_file="../temp_files/3_sampled_decoded_text.txt"
    )

if __name__ == "__main__":

    # main(10000)

    # for vs in range(1000, 20001, 1000):
    #    main(vs)


    for vs in [1000, 2000, 3000, 5000, 8000, 12000]:
        main(vs)