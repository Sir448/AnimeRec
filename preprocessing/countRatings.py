import glob
import os
from collections import Counter
from tqdm import tqdm
import pandas as pd


def count_anime_ratings(filtered_folder: str = "data/filteredRatings") -> Counter:
    """
    Counts how many times each anime_id appears across all filtered CSV files.

    Args:
        filtered_folder (str): Path to the folder containing filtered ratings CSVs.

    Returns:
        Counter: A Counter object mapping anime_id -> occurrence count.
    """
    pattern = os.path.join(filtered_folder, "user_anime????????????_filtered.csv")
    csv_files = glob.glob(pattern)

    anime_counter = Counter()

    for file_path in tqdm(csv_files, desc="Processing CSV files"):
        df = pd.read_csv(file_path)
        df.columns = ["user_id", "anime_id", "score"]
        anime_counter.update(df["anime_id"].values)

    return anime_counter


