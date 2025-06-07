# TODO: Should not be in project, this only serves for getting colormap rgb's
#       for manual plotting in wandb. Please delete done.
#


import numpy as np
import matplotlib.cm as cm

# --- Tweak these parameters ---

# 1. How many distinct colors do you want to generate?
#    (e.g., if you have 8 models, set this to 8)
num_colors = 9

# 2. What are the approximate minimum and maximum sizes of your models?
#    This helps map the colors correctly to your model range.
#    The script spaces the colors evenly on a LOG scale between these two values.
min_model_size = 16  # 75 Million
max_model_size = 2048*2  # 10 Billion

# 3. (Optional) You can change the colormap. 'inferno' is the one from the image.
#    Other good options: 'plasma', 'magma', 'viridis'
colormap_name = 'plasma'

# --- The script ---

# Get the colormap object
cmap = cm.get_cmap(colormap_name)

# Create a list of numbers from 0.0 to 1.0 to sample the colormap
# These represent the "positions" along the color gradient.
color_positions = np.linspace(0, 1, num_colors)

# Generate the colors
# The output is in RGBA format (Red, Green, Blue, Alpha), with values from 0.0 to 1.0
colors_rgba = cmap(color_positions)

# --- Print the results in a user-friendly format ---

# These are the model sizes that each color corresponds to, spaced logarithmically
log_space_sizes = np.logspace(np.log2(min_model_size), np.log2(max_model_size), num=num_colors, base=2)

print(f"--- Generated {num_colors} colors from the '{colormap_name}' colormap ---")
print("The 'Approx. Model Size' shows which model gets which color (smallest to largest).\n")

for i in range(num_colors):
    rgba = colors_rgba[i]

    # Convert from [0, 1] float to [0, 255] integer for standard RGB
    r_int, g_int, b_int = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)

    # Create the HEX code
    hex_code = f"#{r_int:02x}{g_int:02x}{b_int:02x}"

    # Get the corresponding model size for this color
    ctx_size = log_space_sizes[i]
    size_str = f"{int(ctx_size)}"

    print(f"Color {i + 1:2d} (for ~{size_str} ctx):")
    print(f"  - RGB (0-255): ({r_int}, {g_int}, {b_int})")
    print(f"  - HEX:         {hex_code}\n")