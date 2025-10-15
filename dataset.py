import torch
import random
from math import ceil

class RatingsPTDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, pt_files, seed):
        super().__init__()
        self.pt_files = pt_files
        self.seed = seed
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Global RNG (shared seed ensures same shuffle across all workers)
        rng = random.Random(self.seed)

        pt_files = list(self.pt_files)

        # Global shuffle (same order for all workers)
        rng.shuffle(pt_files)

        if worker_info is None:
            assigned_files = pt_files
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            # Deterministic split, no overlap
            files_per_worker = ceil(len(pt_files) / num_workers)
            start = worker_id * files_per_worker
            end = min(start + files_per_worker, len(pt_files))
            assigned_files = pt_files[start:end]

        # Each worker now has a unique set of files
        # Local RNG for within-file shuffling
        local_rng = random.Random(self.seed + (worker_id if worker_info else 0))

        for pt_file in assigned_files:
            data = torch.load(pt_file, map_location="cpu")
            
            user_idx = data["user_idx"]
            anime_idx = data["anime_idx"]
            scores = data["scores"]

            # Generate a list of indices and shuffle them
            indices = list(range(len(user_idx)))
            local_rng.shuffle(indices)

            # Yield samples one by one in shuffled order
            for idx in indices:
                yield user_idx[idx], anime_idx[idx], scores[idx]
