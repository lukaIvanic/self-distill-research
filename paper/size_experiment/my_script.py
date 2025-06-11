import torch
import os
import argparse
import copy
from collections.abc import Mapping, Sequence


# --- Helper Functions ---

def format_size(size_bytes):
    """Formats size in bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes is None:
        return "N/A"
    if size_bytes == 0:
        return "0 Bytes"
    if size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def get_object_info(obj, path='<root>'):
    """
    Recursively traverses an object and returns a flat list of dictionaries,
    each describing a tensor or other element with its path and size.
    """
    results = []

    if isinstance(obj, torch.Tensor):
        size = obj.numel() * obj.element_size()
        results.append({
            'path': path,
            'type': 'Tensor',
            'shape': tuple(obj.shape),
            'dtype': obj.dtype,
            'size_bytes': size,
            'size_str': format_size(size)
        })
    elif isinstance(obj, Mapping):
        # Calculate size of the dictionary structure itself (overhead)
        total_size = 0
        children_info = []
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path != '<root>' else key
            child_info = get_object_info(value, child_path)
            children_info.extend(child_info)
            # Sum up sizes of children
            for info in child_info:
                if info['size_bytes']:
                    total_size += info['size_bytes']

        results.append({
            'path': path,
            'type': f'Dictionary ({type(obj).__name__})',
            'len': len(obj),
            'size_bytes': total_size,
            'size_str': format_size(total_size)
        })
        results.extend(children_info)
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        total_size = 0
        children_info = []
        for i, item in enumerate(obj):
            child_path = f"{path}[{i}]"
            child_info = get_object_info(item, child_path)
            children_info.extend(child_info)
            for info in child_info:
                if info.get('size_bytes'):
                    total_size += info['size_bytes']

        results.append({
            'path': path,
            'type': f'{type(obj).__name__}',
            'len': len(obj),
            'size_bytes': total_size,
            'size_str': format_size(total_size)
        })
        results.extend(children_info)
    else:
        # For non-tensor, non-collection types, size is negligible
        results.append({
            'path': path,
            'type': type(obj).__name__,
            'value': str(obj)[:80],  # Truncate long strings
            'size_bytes': 0,
            'size_str': 'N/A'
        })
    return results


def load_checkpoint(filepath):
    """Loads a checkpoint safely to the CPU."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None
    try:
        print(f"Loading '{os.path.basename(filepath)}'...")
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        return checkpoint
    except Exception as e:
        print(f"An error occurred while loading '{filepath}': {e}")
        return None


# --- Mode Handlers ---

def handle_inspect(args):
    """Handler for the 'inspect' command."""
    print("-" * 80)
    print(f"Inspecting file: {args.filepath}")
    file_size = os.path.getsize(args.filepath)
    print(f"Total file size: {format_size(file_size)}")
    print("-" * 80)

    checkpoint = load_checkpoint(args.filepath)
    if checkpoint is None:
        return

    info_list = get_object_info(checkpoint)
    for info in info_list:
        path = info['path']
        type_info = f"Type: {info['type']}"

        details = []
        if 'len' in info:
            details.append(f"Items: {info['len']}")
        if 'shape' in info:
            details.append(f"Shape: {info['shape']}")
            details.append(f"Dtype: {info['dtype']}")

        size_str = f"Size: {info['size_str']}"

        indent_level = path.count('.') + path.count('[')
        indent = "  " * indent_level

        print(f"{indent}{path:<40} | {type_info:<25} | {size_str:<15} | {', '.join(details)}")


def handle_clean(args):
    """Handler for the 'clean' command."""
    print("-" * 80)
    print(f"Cleaning file: {args.filepath}")
    original_size = os.path.getsize(args.filepath)
    print(f"Original file size: {format_size(original_size)}")
    print("-" * 80)

    checkpoint = load_checkpoint(args.filepath)
    if checkpoint is None: return

    if not isinstance(checkpoint, dict):
        print("Error: The checkpoint root is not a dictionary. Cannot remove keys.")
        return

    keys = list(checkpoint.keys())
    print("Available top-level keys to remove:")
    for i, key in enumerate(keys):
        # Get size info for this specific top-level key
        key_info_list = get_object_info(checkpoint[key])
        key_size = sum(info.get('size_bytes', 0) for info in key_info_list if info.get('size_bytes'))
        print(f"  {i + 1}. {key} ({format_size(key_size)})")

    while True:
        try:
            choice = input("\nEnter the numbers of the keys to remove (e.g., '1 3'), or press Enter to cancel: ")
            if not choice.strip():
                print("Cleaning cancelled.")
                return

            indices_to_remove = [int(x) - 1 for x in choice.split()]
            if not all(0 <= i < len(keys) for i in indices_to_remove):
                raise ValueError("One or more numbers are out of range.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter space-separated numbers corresponding to the keys.")

    keys_to_remove = [keys[i] for i in indices_to_remove]
    print(f"\nThese keys will be removed: {', '.join(keys_to_remove)}")

    # Create a new dictionary without the selected keys
    new_checkpoint = {k: v for k, v in checkpoint.items() if k not in keys_to_remove}

    output_path = args.output
    print(f"Saving cleaned checkpoint to: {output_path}")
    torch.save(new_checkpoint, output_path)

    new_size = os.path.getsize(output_path)
    print(f"\nOriginal size: {format_size(original_size)}")
    print(f"New size:      {format_size(new_size)}")
    print(f"Reduction:     {format_size(original_size - new_size)}")
    print("\nDone.")


def handle_compare(args):
    """Handler for the 'compare' command."""
    file1, file2 = args.files
    print("-" * 80)
    print(f"Comparing:")
    print(f"  File 1: {os.path.basename(file1)} ({format_size(os.path.getsize(file1))})")
    print(f"  File 2: {os.path.basename(file2)} ({format_size(os.path.getsize(file2))})")
    print("-" * 80)

    ckpt1 = load_checkpoint(file1)
    ckpt2 = load_checkpoint(file2)
    if ckpt1 is None or ckpt2 is None: return

    info1 = {item['path']: item for item in get_object_info(ckpt1) if 'shape' in item}
    info2 = {item['path']: item for item in get_object_info(ckpt2) if 'shape' in item}

    all_paths = sorted(list(set(info1.keys()) | set(info2.keys())))

    print(f"{'Path':<40} | {'File 1 Details':<40} | {'File 2 Details':<40}")
    print(f"{'-' * 40} | {'-' * 40} | {'-' * 40}")

    for path in all_paths:
        item1 = info1.get(path)
        item2 = info2.get(path)

        details1, details2 = "N/A", "N/A"
        is_diff = False

        if item1 and item2:
            details1 = f"S:{item1['size_str']}, D:{item1['dtype']}, Shp:{item1['shape']}"
            details2 = f"S:{item2['size_str']}, D:{item2['dtype']}, Shp:{item2['shape']}"
            if item1['size_bytes'] != item2['size_bytes'] or item1['shape'] != item2['shape'] or item1['dtype'] != \
                    item2['dtype']:
                is_diff = True
        elif item1:
            details1 = f"S:{item1['size_str']}, D:{item1['dtype']}, Shp:{item1['shape']}"
            details2 = "--- NOT FOUND ---"
            is_diff = True
        elif item2:
            details1 = "--- NOT FOUND ---"
            details2 = f"S:{item2['size_str']}, D:{item2['dtype']}, Shp:{item2['shape']}"
            is_diff = True

        if is_diff:
            print(f"{path:<40} | {details1:<40} | {details2:<40}  <-- DIFF")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="A tool to inspect, clean, and compare PyTorch checkpoint (.pt, .pth) files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Inspect command
    parser_inspect = subparsers.add_parser('inspect', help='Inspect a single checkpoint file and show its contents.')
    parser_inspect.add_argument('filepath', type=str, help='Path to the .pt file.')
    parser_inspect.set_defaults(func=handle_inspect)

    # Clean command
    parser_clean = subparsers.add_parser('clean',
                                         help='Remove top-level keys from a checkpoint and save it as a new file.')
    parser_clean.add_argument('filepath', type=str, help='Path to the input .pt file.')
    parser_clean.add_argument('-o', '--output', type=str, required=True, help='Path for the new, cleaned output file.')
    parser_clean.set_defaults(func=handle_clean)

    # Compare command
    parser_compare = subparsers.add_parser('compare', help='Compare two checkpoint files side-by-side.')
    parser_compare.add_argument('files', nargs=2, type=str, help='Paths to the two .pt files to compare.')
    parser_compare.set_defaults(func=handle_compare)

    if os.sys.argv.__len__() == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()