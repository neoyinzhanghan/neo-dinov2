import pandas as pd
import matplotlib.pyplot as plt
import os

# Since the training metrics are now known to be in a JSON file, let's read the JSON file into a pandas DataFrame
json_file_path = "/home/bear/Documents/neo/neo-dinov2/log/training_metrics.json"
data = pd.read_json(json_file_path, lines=True)

# Now, we will plot each column against the iteration number
# and save the plots in the same directory as the JSON file
plot_directory = os.path.dirname(json_file_path)

# Create a directory for the plots if it does not exist
os.makedirs(plot_directory, exist_ok=True)

# Iterate over each column except 'iteration' and create a plot for it
for column in data.columns:
    if column != "iteration":
        plt.figure(figsize=(10, 5))
        plt.plot(data["iteration"], data[column], label=column)
        plt.xlabel("Iteration")
        plt.ylabel(column)
        plt.title(f"{column} over Iterations")
        plt.legend()
        plt.grid(True)
        # Save each plot with the name of the metric as the filename
        plot_filename = f"{column}_plot.png"
        plot_path = os.path.join(plot_directory, plot_filename)
        plt.savefig(plot_path)
        plt.close()

# Return the directory where the plots are saved
plot_directory
