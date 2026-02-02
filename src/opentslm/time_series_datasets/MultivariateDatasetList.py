import torch
import numpy as np
import gc
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from datasets import load_dataset
from typing import Optional, List, Dict, Any

DEFAULT_SUBSETS = [
    "traffic", "electricity_15min", "solar_1h", "wiki2000_nips", 
    "m4_hourly", "weatherbench_daily"
]

CHRONOS_PATH = "/scratch/Projects/rl_timeseries/pretraining_data/multivariate/chronos_datasets"

PROMPT = "This is a multivariate time series dataset. The following data contains multiple time series variables recorded over time."


class MemoryEfficientMultivariateDataset(IterableDataset):
    def __init__(
        self, 
        subsets: List[str],
        batch_size: int,
        batches_per_load: int = 100, 
        context_length: int = 512,
        prediction_length: int = 64,
        seed: int = 42
    ):
        self.subsets = subsets
        self.batch_size = batch_size
        self.batches_per_load = batches_per_load
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_seq_len = context_length + prediction_length
        self.seed = seed

    def _load_single_dataset(self, name: str):
        try:
            ds = load_dataset(CHRONOS_PATH, name, split="train", trust_remote_code=True)
            all_series = [np.array(x['target'], dtype=np.float32) for x in ds]
            
            if not all_series: return None
            
            min_len = min(len(s) for s in all_series)
            if min_len < self.total_seq_len:
                return None
                
            matrix = np.stack([s[:min_len] for s in all_series], axis=0).T
            
            del all_series
            return matrix
            
        except Exception as e:
            print(f"Skipping {name}: {e}")
            return None



    def __iter__(self):
        worker_info = get_worker_info()
        rng = np.random.default_rng(self.seed + (worker_info.id if worker_info else 0))
        
        while True:
            shuffled_subsets = list(self.subsets)
            rng.shuffle(shuffled_subsets)
            
            for dataset_name in shuffled_subsets:
                
                matrix = self._load_single_dataset(dataset_name)
                if matrix is None: continue
                
                total_time, num_variates = matrix.shape
                max_start = total_time - self.total_seq_len
                
                for _ in range(self.batches_per_load):
                    start_indices = rng.integers(0, max_start + 1, size=self.batch_size)
                    
                    windows = np.stack([
                        matrix[t : t + self.total_seq_len, :] for t in start_indices
                    ])
                    
                    # ids, loc, scale = self.tokenizer.encode(windows)
                    ids = windows
                    label_ids = ids[:, self.context_length:, :]
                    
                    label_raw = windows[:, self.context_length:, :]

                    batch = {
                        "time_series": torch.tensor(ids[:, :self.context_length, :], dtype=torch.long),
                        "pre_prompt": PROMPT,
                        'answer': None,
                        'post_prompt': None,
                        'time_series_text': None,

                    }

                    ts_batch = torch.tensor(ids[:, :self.context_length, :], dtype=torch.long)
                    batch = [
                        {
                            "time_series": ts_batch[i],
                            "pre_prompt": PROMPT,
                            'answer': None,
                            'post_prompt': None,
                            'time_series_text': None,
                        }
                        for i in range(ts_batch.shape[0])
                    ]
                    
                    yield batch
                
                del matrix
                gc.collect()

def get_chronos_dataloader(
    subsets: List[str] = DEFAULT_SUBSETS,
    batch_size: int = 32,
    num_workers: int = 4,
    batches_per_load: int = 50 
):
    dataset = MemoryEfficientMultivariateDataset(
        subsets=subsets,
        batch_size=batch_size,
        batches_per_load=batches_per_load
    )
    
    return DataLoader(
        dataset, 
        batch_size=None, 
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    loader = get_chronos_dataloader(
        subsets=["nn5", "electricity_15min"], 
        batch_size=4,
        batches_per_load=2, 
        num_workers=0
    )
    
    print("Starting Loop...")
    for i, batch in enumerate(loader):
        print('batch: ', len(batch))
        print('batch[0] keys: ', batch[0])
        if i >= 3: break