import os
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LinearSegmentedColormap

# --- Configuration ---

DATASETS = ['1M_org', '5M']
DATASET_SHOW = '5M'

USE_TRAIN_DATA = False

DATA_DIRECTORY = "./paper/01_1M_ctx_len_exp/01_graph_experiments/01_raw_datas/"
if DATASET_SHOW == '1M_org':
    DATA_DIRECTORY = DATA_DIRECTORY
elif DATASET_SHOW == '5M':
    DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, '02_5M_datas')


FILE_PATTERN = "0*_max_ctx_*.json"
CONDENSE = False # Condenses for eg. 1024-2047 sub-buckets of len 64,
                # to one averaged out bucket, (in general)

# --- Colormap Configuration (to match our previous discussion) ---
# This creates a "clipped" version of 'inferno' to avoid extreme darks/brights
CLIP_START = 0.15  # Clips 15% from the dark bottom
CLIP_END = 0.9  # Clips 10% from the bright top


def create_dummy_data(directory):
    """Generates dummy JSON files if the directory is empty, for testing."""
    print(f"Warning: Directory '{directory}' not found or empty. Creating dummy data for demonstration.")
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(6, 12):  # 2^6=64 to 2^11=2048
        max_ctx = 2 ** i
        filename = os.path.join(directory, f"01_max_ctx_{max_ctx}.json")
        data = {}
        base_loss = 4.0 - 0.2 * (i - 6)
        data["train_loss"] = base_loss + np.random.rand() * 0.1

        for j in range(max_ctx // 64):
            start = j * 64
            end = start + 63
            loss_key = f"loss_ctx_{start}_{end}"
            # Make loss slightly higher for later contexts
            data[loss_key] = base_loss + (j / (max_ctx // 64)) * 0.5 + np.random.rand() * 0.2

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    print("Dummy data created. Please run the script again.")


def parse_run_data(filepath):
    """Parses a single JSON file to extract relevant loss data."""
    # Extract max_ctx from filename using regex
    print(filepath)
    print(os.path.basename(filepath))
    match = re.search(r'max_ctx_(\d+)_(\d+)K', os.path.basename(filepath))
    if not match:
        return None
    max_ctx = int(match.group(1))
    batch_size = int(match.group(2))
    print(max_ctx, batch_size)
    with open(filepath, 'r') as f:
        raw_data = json.load(f)

    if USE_TRAIN_DATA:
        avg_loss = raw_data.get("train_loss")
    else:
        avg_loss = raw_data.get("avg_validation_loss")

    print(avg_loss)
    bucketed_avg_losses = {}
    # Extract bucketed losses
    for key, value in raw_data.items():
        if USE_TRAIN_DATA:
            bucket_match = re.match(r'loss_ctx_(\d+)_(\d+)', key)
        else:
            bucket_match = re.match(r'avg_loss_ctx_(\d+)_(\d+)', key)

        if bucket_match:
            start_token = int(bucket_match.group(1))
            bucketed_avg_losses[start_token] = value

    print(bucketed_avg_losses)
    if not avg_loss or not bucketed_avg_losses:
        return None

    return {
        "max_ctx": max_ctx,
        'batch_size': f"{batch_size}K",
        "avg_loss": avg_loss,
        "bucketed_avg_losses": bucketed_avg_losses
    }

def condense_if_enabled(buckets):
    if not CONDENSE:
        return buckets

    if len(buckets) <= 2:
        return buckets

    new_buckets = buckets[:2]  # for 0-64 and 64-128 buckets
    for i in range(int(np.log2(len(buckets)))-1):
        new_buckets.append((2**(i+8), np.mean([x[1] for x in buckets[2**(i+1):2**(i+2)]])))
        #print(2**(i+8))
        #print(buckets[2**(i+1):2**(i+2)])

    #print(new_buckets)
    return new_buckets


def print_loss_comparison_table(all_runs_buckets: dict):
    """
    Prints a beautifully formatted comparison table of loss values.

    The table shows aggregated loss buckets in rows and different
    max context length experiments in columns.

    Args:
        all_runs_buckets (dict): A dictionary where keys are run identifiers
            (e.g., 'ctx_128', 'ctx_2048') and values are the list of
            bucket tuples [(start, loss), ...].
    """
    if not all_runs_buckets:
        print("No data provided to generate the table.")
        return

    # --- 1. GATHER AND STRUCTURE DATA ---

    # Sort run keys numerically (128, 256, 512...) instead of alphabetically
    run_keys = sorted(all_runs_buckets.keys(), key=lambda k: int(re.search(r'\d+', k).group()))

    # Create a fast lookup map: {run_key: {bucket_start: loss}}
    data_map = {key: dict(buckets) for key, buckets in all_runs_buckets.items()}

    # Find all unique bucket start positions across all runs to define the rows
    all_bucket_starts = sorted(list(set(
        start for buckets in all_runs_buckets.values() for start, loss in buckets
    )))

    col_minimums = {}
    for run_key in run_keys:
        losses = [loss for loss in data_map[run_key].values() if loss is not None]
        col_minimums[run_key] = min(losses) if losses else None

    # --- 2. PREPARE FORMATTING ---

    # Helper to calculate the end of an exponentially growing bucket
    def get_bucket_end(start):
        return int(2**(np.log2(start+64)))

    # Determine column headers and calculate required widths
    headers = ["Bucket Range"] + [f"Max Ctx {k.split('_')[-1]}" for k in run_keys]


    # Calculate column widths for perfect alignment
    col_widths = [len(h) for h in headers]
    for i, start in enumerate(all_bucket_starts):
        # Update width for the first column (bucket labels)
        label = f"{start}-{get_bucket_end(start)}"
        col_widths[0] = max(col_widths[0], len(label))

        # Data columns have a fixed width based on header or "xx.xxxx" format
        for j, run_key in enumerate(run_keys):
            col_widths[j + 1] = max(col_widths[j + 1], 8)  # Accommodates "xx.xxxx"

    # --- 3. PRINT THE TABLE ---

    # Print Header
    header_str = " | ".join([h.center(col_widths[i]) for i, h in enumerate(headers)])
    separator_str = "-+-".join(["-" * w for w in col_widths])

    print("\n" + "=" * len(separator_str))
    print("           Aggregated Loss Comparison by Context Window")
    print("=" * len(separator_str))
    print(header_str)
    print(separator_str)

    # Print Body
    BOLD = '\033[1m'
    GREEN = '\033[92m'  # For row minimums
    MAGENTA = '\033[95m'  # For column minimums
    BG_YELLOW = '\033[43m'  # For row AND column minimums
    RESET = '\033[0m'
    # Print Body
    for start in all_bucket_starts:
        row_losses = [data_map[run_key].get(start) for run_key in run_keys if data_map[run_key].get(start) is not None]
        min_loss_for_row = min(row_losses) if row_losses else None

        row_list = [f"{start}-{get_bucket_end(start)}".ljust(col_widths[0])]

        for i, run_key in enumerate(run_keys):
            loss = data_map[run_key].get(start)
            min_loss_for_col = col_minimums.get(run_key)
            width = col_widths[i + 1]
            formatted_cell = "-".center(width)

            if loss is not None:
                is_row_min = (loss == min_loss_for_row)
                is_col_min = (loss == min_loss_for_col)
                value_str = f"{loss:.4f}"

                # Apply highlights based on priority
                if is_row_min and is_col_min:
                    style = f"{BG_YELLOW}{MAGENTA}"
                elif is_row_min:
                    style = f"{BOLD}{GREEN}"
                elif is_col_min:
                    style = f"{BOLD}{MAGENTA}"
                else:
                    style = ""

                if style:
                    padding = width - len(value_str)
                    formatted_cell = f'{" " * (padding // 2 + (1*(padding%2==1)))}{style}{value_str}{RESET}{" " * (padding - (padding // 2 + (1*(padding%2==1))))}'
                else:
                    formatted_cell = value_str.center(width)

            row_list.append(formatted_cell)
        print(" | ".join(row_list))

    print("-" * len(separator_str))
    print(f" {BOLD}{GREEN}Green{RESET}: Best model for this context bucket (Row Min)")
    print(f" {BOLD}{MAGENTA}Magenta{RESET}:  Easiest bucket for this model (Column Min)")
    print(f" {BG_YELLOW}{MAGENTA}Yellow BG{RESET}: Minimum of both its row and column")
    print()





def main():
    """Main function to find data, parse it, and generate the plot."""

    filepaths = glob.glob(os.path.join(DATA_DIRECTORY, FILE_PATTERN))

    if not filepaths:
        create_dummy_data(DATA_DIRECTORY)
        return


    print(f"Found files: {filepaths}")
    # 1. Read and parse all data files
    all_runs_data = []
    for fp in filepaths:
        data = parse_run_data(fp)
        if data:
            all_runs_data.append(data)

    if not all_runs_data:
        print("No valid data files found or parsed. Exiting.")
        return


    # Sort runs by max_ctx to ensure colors are applied in order
    all_runs_data.sort(key=lambda x: x['max_ctx'])

    # 2. Prepare the custom colormap
    min_ctx = min(run['max_ctx'] for run in all_runs_data)
    max_ctx = max(run['max_ctx'] for run in all_runs_data)

    original_cmap = plt.get_cmap('inferno')
    sliced_colors = original_cmap(np.linspace(CLIP_START, CLIP_END, 256))
    custom_cmap = LinearSegmentedColormap.from_list("custom_inferno", sliced_colors)

    # Use a logarithmic scale for the color mapping, as context sizes are powers of 2
    norm = Normalize(vmin=np.log2(min_ctx), vmax=np.log2(max_ctx))

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    all_buckets = {}



    for run_data in all_runs_data:
        max_ctx_val = run_data['max_ctx']
        batch_size_tokens = run_data['batch_size']
        color = custom_cmap(norm(np.log2(max_ctx_val)))

        ax.axhline(
            y=run_data['avg_loss'],
            color=color,
            linestyle='--',
            alpha=0.8,
            linewidth=1.5
        )

        # Plot the bucketed losses
        buckets = sorted(run_data['bucketed_avg_losses'].items())
        buckets = condense_if_enabled(buckets)


        all_buckets[str(max_ctx_val)] = buckets


        x_vals = [b[0]+64 for b in buckets]  # Context start position
        y_vals = [b[1] for b in buckets]  # Loss



        ax.plot(
            x_vals,
            y_vals,
            marker='o',
            markersize=8,
            linestyle='-',
            color=color,
            label=f'Max Ctx = {max_ctx_val} ({batch_size_tokens})'
        )

        # 4. Finalize and show the plot
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    print(all_buckets)
    print_loss_comparison_table(all_buckets)



    # ax.set_title("Loss by Context Window Across Different Max Context Lengths", fontsize=16)
    ax.set_xlabel("Context Window", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title="Experiment")

    # Optional: Use a log scale for the x-axis if it helps visualization
    #ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(f"./paper/01_1M_ctx_len_exp/01_graph_experiments/loss_by_context_window_{DATASET_SHOW}.png", dpi=300)
    print(f"Plot saved as 'loss_by_context_window_{DATASET_SHOW}.png'")
    plt.show()


if __name__ == '__main__':
    main()