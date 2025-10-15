import os
import glob
import torch
import pandas as pd
from tqdm import tqdm


def build_user_anime_tensors(threshold: int = 500) -> None:
    """
    Converts filtered user-anime CSV files into PyTorch tensor files.
    
    Args:
        threshold (int): The rating threshold used in the dataset folder names.
    """
    # Input and output paths
    csv_pattern = f"data/filteredRatingsThreshold{threshold}/user_anime????????????_filtered{threshold}.csv"
    output_dir = f"data/pt_files{threshold}"
    os.makedirs(output_dir, exist_ok=True)

    # Load mapping and genre data
    anime_id_to_idx = torch.load(f"data/pt_files{threshold}/anime_id_to_idx.pt", weights_only=False)

    # Collect all user IDs from CSVs
    csv_files = glob.glob(csv_pattern)
    user_set = set()
    for file_path in tqdm(csv_files, desc="Collecting user IDs"):
        df = pd.read_csv(file_path)
        df.columns = ["user_id", "anime_id", "score"]
        user_set.update(df["user_id"].unique())

    # Build mapping from user_id to index
    user_ids = list(user_set)
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    # Save user_id_to_idx mapping
    torch.save(user_id_to_idx, os.path.join(output_dir, "user_id_to_idx.pt"))

    # Convert and save each CSV to tensor format
    for i, file_path in tqdm(enumerate(csv_files), total=len(csv_files), desc="Converting CSVs"):
        df = pd.read_csv(file_path)

        user_idx = torch.tensor([user_id_to_idx[u] for u in df.user_id], dtype=torch.long)
        anime_idx = torch.tensor([anime_id_to_idx[a] for a in df.anime_id], dtype=torch.long)
        scores = torch.tensor(df.score.values, dtype=torch.float32)

        torch.save(
            {
                "user_idx": user_idx,
                "anime_idx": anime_idx,
                "scores": scores,
            },
            os.path.join(output_dir, f"user_anime{i:012d}_filtered{threshold}.pt"),
        )

    print("✅ Conversion complete!")


if __name__ == "__main__":
    build_user_anime_tensors(threshold=500)
