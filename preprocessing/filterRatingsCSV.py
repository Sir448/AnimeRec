import pandas as pd
import glob
import os


def filter_ratings_csvs(
    original_folder="data/original",
    filtered_folder="data/filteredRatings",
    file_pattern="user_anime????????????.csv"
):
    """
    Process all user-anime CSV files:
    1. Filter out rows with missing scores.
    2. Keep only ['user_id', 'anime_id', 'score'] columns.
    3. Convert scores to integers.
    4. Save processed files to filtered_folder with "_filtered" suffix.
    
    Args:
        original_folder (str): Path to folder with raw CSV files.
        filtered_folder (str): Path to folder where processed CSVs will be saved.
    """

    # Make sure output folder exists
    os.makedirs(filtered_folder, exist_ok=True)

    # Step 1: Find all CSVs matching pattern "user_animeXXXXXXXXXXXX.csv"
    pattern = os.path.join(original_folder, file_pattern)
    csv_files = glob.glob(pattern)

    for file_path in csv_files:
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        new_name = f"{name}_filtered{ext}"
        output_path = os.path.join(filtered_folder, new_name)

        # Step 2: Read CSV and filter missing scores
        df = pd.read_csv(file_path, sep="\t")
        df = df[df["score"].notna()]

        # Step 3: Keep only desired columns
        df = df[['user_id', 'anime_id', 'score']]

        # Step 4: Convert score to integer
        df['score'] = df['score'].astype(int)

        # Step 5: Save filtered DataFrame
        df.to_csv(output_path, index=False)

        print(f"Processed {base_name} → {new_name}")

if __name__ == "__main__":
    filter_ratings_csvs()