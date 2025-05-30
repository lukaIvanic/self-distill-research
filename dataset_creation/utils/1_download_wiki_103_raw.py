from datasets import load_dataset
import os


def download_wikitext103_raw(save_path):
    """
    Downloads the wikitext-103-raw-v1 dataset using the Hugging Face datasets library
    and saves the main_space, validation, and test splits to separate text files.

    Args:
        save_path (str): The directory where the text files will be saved.
    """
    print("Starting download of wikitext-103-raw-v1...")
    try:
        # Load the raw version of wikitext-103
        # This configuration provides the text before <unk> replacement
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        return

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    print(f"Data will be saved in: {os.path.abspath(save_path)}")

    # The dataset is a DatasetDict containing train, validation, and test splits
    # Each split has a 'text' column where each entry is a string (often a paragraph or line)

    for split_name in dataset.keys():
        print(f"\nProcessing {split_name} split...")
        file_path = os.path.join(save_path, f"wikitext-103-raw-{split_name}.txt")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # Iterate over each example in the split
                # Each example in wikitext is a dictionary, and dataset[split_name]['text'] is a list of strings
                count = 0
                for example_text in dataset[split_name]['text']:
                    if example_text.strip():  # Write non-empty lines
                        f.write(example_text + "\n")
                        count += 1
            print(f"Successfully saved {count} lines from '{split_name}' split to {file_path}")
        except Exception as e:
            print(f"Error saving {split_name} split: {e}")

    print("\nAll splits processed.")
    print("The dataset typically contains:")
    for split_name in dataset.keys():
        print(f" - {split_name}: {len(dataset[split_name])} examples (lines/paragraphs)")


if __name__ == "__main__":
    # Define where you want to save the text files
    # Your tokenizer script can then use these files as input
    output_directory = "../wikitext103_raw_corpus2"
    download_wikitext103_raw(save_path=output_directory)

    # Example of how your tokenizer script (tng_02_preprocess_new.py) might use this:
    # You would set `input_text_file` in your script to something like:
    # input_text_file = "wikitext103_raw_corpus/wikitext-103-raw-train.txt"
    print(f"\nTo use this with your script, set 'input_text_file' in 'tng_02_preprocess_new.py'")
    print(f"For example, for the main_space data:")
    print(f"input_text_file = \"{os.path.join(output_directory, 'wikitext-103-raw-train.txt')}\"")

