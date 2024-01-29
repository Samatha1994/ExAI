import pandas as pd
import os

def analyze_activations(csv_file_path, output_dir):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Analyze each column (except the last one which contains filenames)
    positive_images = {}
    negative_images = {}

    for column in data.columns[:-1]:  # Exclude the last column (filenames)
        highest_value = data[column].max()
        threshold_high = 0.80 * highest_value
        threshold_low = 0.20 * highest_value

        # Get filenames for positive images (above 80% of highest value)
        positive_images[column] = data[(data[column] >= threshold_high) & (data[column] != 0)]['filenames'].tolist()

        # Get filenames for negative images (below 20% of highest value)
        negative_images[column] = data[(data[column] <= threshold_low) & (data[column] != 0)]['filenames'].tolist()

    # Saving the results
    save_results(positive_images, os.path.join(output_dir, "positive_images.csv"))
    save_results(negative_images, os.path.join(output_dir, "negative_images.csv"))

def save_results(data_dict, file_path):
    # Convert dictionary to DataFrame for saving
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in data_dict.items()]))
    df.to_csv(file_path, index=False)

