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

PROMPT = "This is a multivariate time series dataset. The following data contains multiple time series variables recorded over time. Please make a prediction for the next {n_pred} time steps based on the previous {n_context} time steps."
class ChronosTokenizer:
    """
    Multivariate-aware Tokenizer.
    """
    def __init__(self, vocab_size: int = 4096, special_tokens: int = 2):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens 
        self.num_bins = vocab_size - special_tokens
        self.centers = np.linspace(-3, 3, self.num_bins)
        self.bin_edges = (self.centers[1:] + self.centers[:-1]) / 2

    def encode(self, x: np.ndarray):
        if x.size == 0: return x, 0.0, 1.0
        
        loc = np.nanmean(x, axis=-2, keepdims=True)
        scale = np.nanstd(x, axis=-2, keepdims=True)
        
        scale = np.where((scale == 0) | (np.isnan(scale)), 1.0, scale)
        loc = np.where(np.isnan(loc), 0.0, loc)
        
        x_scaled = (x - loc) / scale
        
        ids = np.digitize(x_scaled, self.bin_edges) + self.special_tokens
        ids = np.clip(ids, self.special_tokens, self.vocab_size - 1)
        
        return ids, loc, scale

class MemoryEfficientMultivariateDataset(IterableDataset):
    def __init__(
        self, 
        subsets: List[str],
        batch_size: int,
        batches_per_load: int = 100, 
        context_length: int = 512,
        prediction_length: int = 64,
        tokenizer: Optional[ChronosTokenizer] = None,
        seed: int = 42
    ):
        self.subsets = subsets
        self.batch_size = batch_size
        self.batches_per_load = batches_per_load
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_seq_len = context_length + prediction_length
        self.tokenizer = tokenizer if tokenizer else ChronosTokenizer()
        self.seed = seed

    def _load_single_dataset(self, name: str):
        """
        Loads ONE dataset entirely into RAM.
        Returns: matrix [Time, Variates] or None if failed.
        """
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
                    
                    ids, loc, scale = self.tokenizer.encode(windows)
                    label_ids = ids[:, self.context_length:, :]
                    
                    label_raw = windows[:, self.context_length:, :]

                    batch = {
                        "input_ids": torch.tensor(ids[:, :self.context_length, :], dtype=torch.long),
                        "labels":    torch.tensor(label_ids, dtype=torch.long),
                        "labels_raw": torch.tensor(label_raw, dtype=torch.float32),
                        "scale":     torch.tensor(scale, dtype=torch.float32),
                        "loc":       torch.tensor(loc, dtype=torch.float32),
                        "dataset_name": dataset_name,
                        "prompt": PROMPT.format(
                            n_context=self.context_length, n_pred=self.prediction_length
                        )
                    }
                    
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
        print(f"\nBatch {i} from {batch['dataset_name']}:")
        print(f"Input IDs (Encoder): {batch['input_ids'].shape}")  
        print(f"Labels    (Decoder): {batch['labels'].shape}")     
        print(f"Raw Float (Metric):  {batch['labels_raw'].shape}") 
        
        print(f"Sample Token: {batch['labels'][0, 0, 0]}")
        print(f"Sample Float: {batch['labels_raw'][0, 0, 0]:.4f}")
        
        if i >= 3: break