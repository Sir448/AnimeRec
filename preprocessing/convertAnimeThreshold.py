import os
import torch
import pandas as pd
from collections import Counter

def save_thresholded_anime_pt_files(
    threshold=500,
    anime_counter=None,
    base_data_dir="data"
):
    """
    Filters 'anime_filtered.csv' based on a rating count threshold,
    and saves corresponding PyTorch tensor files (.pt) for IDs and genres.

    Args:
        threshold (int): Minimum number of ratings required for an anime to be kept.
        anime_counter (Counter): Precomputed Counter mapping anime_id → rating count.
                                 Required to determine which animes meet the threshold.
        base_data_dir (str): Base data directory containing processed CSVs.
    """
    if anime_counter is None:
        from countRatings import count_anime_ratings
        
        anime_counter = count_anime_ratings()

    # Define folders
    data_dir = os.path.join(base_data_dir, f"filteredRatingsThreshold{threshold}")
    pt_dir = os.path.join(base_data_dir, f"pt_files{threshold}")
    os.makedirs(pt_dir, exist_ok=True)

    # Load filtered anime CSV
    anime_csv_path = os.path.join(data_dir, "anime_filtered.csv")
    if not os.path.exists(anime_csv_path):
        raise FileNotFoundError(f"Could not find {anime_csv_path}")

    anime_df = pd.read_csv(anime_csv_path)

    # Apply threshold filter
    anime_df = anime_df[anime_df['anime_id'].map(anime_counter) >= threshold]

    # Save the filtered version (overwrite existing)
    anime_df.to_csv(anime_csv_path, index=False)

    # Generate id-to-index mapping
    anime_ids = anime_df['anime_id'].values
    anime_id_to_idx = {aid: idx for idx, aid in enumerate(anime_ids)}

    # Save mapping
    torch.save(anime_id_to_idx, os.path.join(pt_dir, "anime_id_to_idx.pt"))

    # Extract genre matrix
    anime_genres = torch.tensor(anime_df.iloc[:, 2:].values, dtype=torch.float32)

    # Save genre tensor
    torch.save(anime_genres, os.path.join(pt_dir, "anime_genres.pt"))

    print(f"Saved .pt files for anime IDs and genres (threshold = {threshold}) at {pt_dir}")


# --- Main guard for standalone execution ---
if __name__ == "__main__":
    save_thresholded_anime_pt_files(threshold=500)
