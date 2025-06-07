# TODO: see todo of 01_temp_color_utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import wandb

# --- Parameters (MUST match your color generation script) ---
min_model_size = 64      # 75 Million
max_model_size = 2048   # 10 Billion
#    Other good options: 'inferno' 'plasma', 'magma', 'viridis'
colormap_name = 'plasma'

# --- The script to generate the legend image ---

# 1. Create a figure and an axis for the colorbar
# We make it tall and thin to look like the legend in the paper
fig, ax = plt.subplots(figsize=(1.5, 6))
fig.subplots_adjust(right=0.5) # Adjust spacing

# 2. Set up the same colormap and normalization as before
cmap = plt.get_cmap(colormap_name)
norm = Normalize(vmin=np.log2(min_model_size), vmax=np.log2(max_model_size))

# 3. Define the ticks and labels for the colorbar
# These are the specific model sizes you want to label on the legend
cbar_ticks_values = np.array([64, 128, 256, 512])
# cbar_ticks_values = np.array([512, 1024, 2048])
cbar_ticks_log = np.log2(cbar_ticks_values)
cbar_tick_labels = [str(x) for x in cbar_ticks_values]

# 4. Create the colorbar itself
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) # This is a required step

cbar = fig.colorbar(sm, cax=ax, orientation='vertical', ticks=cbar_ticks_log)
cbar.ax.set_yticklabels(cbar_tick_labels, fontsize=18)
# cbar.set_label("Max context length", fontsize=20, labelpad=15)

# 5. Save and log the legend image to wandb
# You can log this to any run in your project, or a dedicated "assets" run.
legend_image_path = f"../02_graphs/03_1M_up_to_512_legend.png"
# legend_image_path = f"../../paper/01_1M_ctx_len_exp/02_graphs/04_1M_from_512_legend.png"
plt.savefig(legend_image_path, bbox_inches='tight', dpi=150)
plt.close()

# Log to a new wandb run
run = wandb.init(project="model-visualization-demo", name="assets_and_legends")
run.log({"model_size_legend": wandb.Image(legend_image_path)})
run.finish()

print(f"Legend image saved as '{legend_image_path}' and logged to wandb.")