import pandas as pd
import os

def format_anime_csv(
    input_file="data/original/anime.csv",
    output_folder="data/filteredRatings",
    output_file="anime_filtered.csv"
):
    """
    Process the anime CSV to:
    1. Keep only ['anime_id', 'title', 'genres'] columns.
    2. Convert 'genres' column into multi-hot encoding columns.
    3. Save the processed DataFrame to output_folder/output_file.
    
    Args:
        input_file (str): Path to the raw anime CSV.
        output_folder (str): Folder to save the processed CSV.
        output_file (str): Name of the processed CSV file.
    """
    
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read CSV
    df = pd.read_csv(input_file, sep="\t")

    # Keep only desired columns
    df_filtered = df[['anime_id', 'title', 'genres']]

    # Create multi-hot columns for genres
    genre_multi_hot = df_filtered["genres"].str.get_dummies(sep="|")

    # Combine with anime_id and title
    df_multi_hot = pd.concat([df_filtered[["anime_id", "title"]], genre_multi_hot], axis=1)

    # Save to CSV
    output_path = os.path.join(output_folder, output_file)
    df_multi_hot.to_csv(output_path, index=False)

    print(f"Processed genres saved to {output_path}")

if __name__ == "__main__":
    format_anime_csv()