from preprocessing import *

# Initial processing then convert to PT files
format_anime_csv()
filter_ratings_csvs()
save_anime_pt_files()
save_user_rating_pt_files()

# Count anime ratings
anime_counter = count_anime_ratings()

# Filter anime by rating count threshold
filter_anime_by_threshold(threshold=100, anime_counter=anime_counter)
save_thresholded_anime_pt_files(threshold=100,anime_counter=anime_counter)
build_user_anime_tensors(threshold=100)

filter_anime_by_threshold(threshold=500, anime_counter=anime_counter)
save_thresholded_anime_pt_files(threshold=500,anime_counter=anime_counter)
build_user_anime_tensors(threshold=500)