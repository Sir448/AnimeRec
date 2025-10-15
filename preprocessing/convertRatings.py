import torch
import pandas as pd
import glob
import os
from tqdm import tqdm

def save_user_rating_pt_files(
    filtered_folder="data/filteredRatings",
    pt_dir="data/pt_files",
    anime_id_to_idx_file="data/pt_files/anime_id_to_idx.pt",
    anime_genres_file="data/pt_files/anime_genres.pt"
):
    """
    Convert user-anime CSV ratings into PyTorch tensors and save as .pt files.
    Also generates a user_id_to_idx mapping.

    Args:
        filtered_folder (str): Folder containing filtered user_anime CSV files.
        pt_dir (str): Directory to save the .pt files.
        anime_id_to_idx_file (str): Path to saved anime_id_to_idx.pt file.
        anime_genres_file (str): Path to saved anime_genres.pt file.
    """

    os.makedirs(pt_dir, exist_ok=True)

    # Load anime mappings and genres
    anime_id_to_idx = torch.load(anime_id_to_idx_file, weights_only=False)
    anime_genres = torch.load(anime_genres_file)

    # Find all filtered CSVs
    pattern = os.path.join(filtered_folder, "user_anime????????????_filtered.csv")
    csv_files = glob.glob(pattern)

    # Build user_id mapping
    user_set = set()
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df.columns = ["user_id", "anime_id", "score"]
        user_set.update(df["user_id"].unique())

    user_ids = list(user_set)
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    # Save user mapping
    torch.save(user_id_to_idx, os.path.join(pt_dir, "user_id_to_idx.pt"))

    # Convert CSVs to .pt tensors
    for i, file_path in tqdm(enumerate(csv_files), total=len(csv_files), desc="Processing user ratings"):
        df = pd.read_csv(file_path)
        user_idx = torch.tensor([user_id_to_idx[u] for u in df.user_id], dtype=torch.long)
        anime_idx = torch.tensor([anime_id_to_idx[a] for a in df.anime_id], dtype=torch.long)
        scores = torch.tensor(df.score.values, dtype=torch.float32)

        torch.save({
            "user_idx": user_idx,
            "anime_idx": anime_idx,
            "scores": scores
        }, os.path.join(pt_dir, f"user_anime{i:012d}_filtered.pt"))

    print(f"Saved {len(csv_files)} user rating .pt files and user_id_to_idx mapping to {pt_dir}")


# --- Main guard for standalone execution ---
if __name__ == "__main__":
    save_user_rating_pt_files()
