import torch
import pandas as pd
import os

def save_anime_pt_files(
    filtered_csv="data/filteredRatings/anime_filtered.csv",
    pt_dir="data/pt_files"
):
    """
    Load filtered anime CSV and save:
    1. anime_id_to_idx mapping as a .pt file
    2. genre matrix as a .pt tensor

    Args:
        filtered_csv (str): Path to the filtered anime CSV.
        pt_dir (str): Directory to save the .pt files.
    """
    
    # Make sure output folder exists
    os.makedirs(pt_dir, exist_ok=True)

    # Load filtered CSV
    anime_df = pd.read_csv(filtered_csv)

    # Generate id-to-index mapping
    anime_ids = anime_df['anime_id'].values
    anime_id_to_idx = {aid: idx for idx, aid in enumerate(anime_ids)}

    # Save mapping
    torch.save(anime_id_to_idx, os.path.join(pt_dir, "anime_id_to_idx.pt"))

    # Extract genre matrix (assuming columns 2: are multi-hot genres)
    anime_genres = torch.tensor(anime_df.iloc[:, 2:].values, dtype=torch.float32)

    # Save genres tensor
    torch.save(anime_genres, os.path.join(pt_dir, "anime_genres.pt"))

    print(f"Saved .pt files to {pt_dir}.")


# Main guard for standalone execution
if __name__ == "__main__":
    save_anime_pt_files()
