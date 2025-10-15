import pandas as pd
import glob
import os
from tqdm import tqdm
from collections import Counter

def filter_anime_by_threshold(
    filtered_folder="data/filteredRatings",
    threshold=500,
    output_parent="data",
    anime_counter=None
):
    """
    Filters anime data by minimum number of user ratings (threshold).
    Only keeps anime that appear at least `threshold` times across all user files.

    Args:
        filtered_folder (str): Path to folder with user_anime*_filtered.csv files.
        threshold (int): Minimum number of ratings required for an anime to be kept.
        output_parent (str): Base output directory for threshold-filtered results.

    Output:
        Creates a new folder 'data/filteredRatingsThreshold{threshold}' containing
        filtered CSVs with only the animes meeting the threshold.
    """
    
    # Find all user-anime CSVs
    pattern = os.path.join(filtered_folder, "user_anime????????????_filtered.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found in {filtered_folder}.")
        return

    # Count anime occurrences across all users
    if not anime_counter:
        anime_counter = Counter()
        for file_path in tqdm(csv_files, desc="Counting anime frequencies"):
            df = pd.read_csv(file_path)
            df.columns = ["user_id", "anime_id", "score"]
            anime_counter.update(df["anime_id"].values)

    # Keep only anime with ratings >= threshold
    animes_to_keep = {aid for aid, count in anime_counter.items() if count >= threshold}
    print(f"{len(animes_to_keep)} anime passed the threshold of {threshold} ratings.")

    # Create output folder
    threshold_folder = os.path.join(output_parent, f"filteredRatingsThreshold{threshold}")
    os.makedirs(threshold_folder, exist_ok=True)

    # Filter and save new CSVs
    for file_path in tqdm(csv_files, desc=f"Filtering anime with ≥{threshold} ratings"):
        df = pd.read_csv(file_path)
        df = df[df['anime_id'].isin(animes_to_keep)]

        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        new_name = f"{name}{threshold}{ext}"
        output_path = os.path.join(threshold_folder, new_name)

        df.to_csv(output_path, index=False)

    print(f"Filtered CSVs saved to: {threshold_folder}")


# --- Main guard for standalone execution ---
if __name__ == "__main__":
    filter_anime_by_threshold(threshold=500)
