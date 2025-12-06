import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class DeepGestureDataset(Dataset):
    """
    A PyTorch Dataset optimized for GPU memory usage on high-memory GPUs like the RTX 3090.
    This implementation prefetches data to GPU memory for maximum performance.
    """

    def __init__(self, h5_path, max_gpu_chunks=4):
        """
        Initialize the dataset with direct GPU memory mapping.

        Args:
            h5_path (str): Path to the HDF5 file
            max_gpu_chunks (int): Number of chunks to keep in GPU memory (adjust based on dataset size)
        """
        # Load the entire dataset into CPU memory first
        with h5py.File(h5_path, 'r') as f:
            # Convert to torch tensors immediately
            self.cpu_data = torch.from_numpy(f['data'][:]).float()
            self.cpu_labels = torch.from_numpy(f['labels'][:]).long()

        self.dataset_size = len(self.cpu_data)

        # Calculate optimal chunk size based on 3090's 24GB memory
        # Reserve 4GB for model and other operations
        available_gpu_mem = 20 * 1024 * 1024 * 1024  # 20GB in bytes
        single_sample_size = self.cpu_data[0].element_size() * self.cpu_data[0].nelement()
        self.chunk_size = min(
            self.dataset_size,
            (available_gpu_mem // (single_sample_size * max_gpu_chunks))
        )

        # Initialize GPU memory cache
        self.gpu_cache = {}
        self.cache_order = []
        self.max_gpu_chunks = max_gpu_chunks

    def _load_chunk_to_gpu(self, chunk_idx):
        """
        Load a chunk of data directly to GPU memory.
        """
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.dataset_size)

        # If GPU cache is full, remove oldest chunk
        if len(self.gpu_cache) >= self.max_gpu_chunks:
            oldest_chunk = self.cache_order.pop(0)
            del self.gpu_cache[oldest_chunk]

        # Load new chunk to GPU
        self.gpu_cache[chunk_idx] = {
            'data': self.cpu_data[start_idx:end_idx].cuda(non_blocking=True),
            'labels': self.cpu_labels[start_idx:end_idx].cuda(non_blocking=True)
        }
        self.cache_order.append(chunk_idx)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Get a sample from the dataset, ensuring it's loaded in GPU memory.
        """
        chunk_idx = idx // self.chunk_size
        chunk_offset = idx % self.chunk_size

        # Load chunk to GPU if not already cached
        if chunk_idx not in self.gpu_cache:
            self._load_chunk_to_gpu(chunk_idx)

        # Get data directly from GPU cache
        return (
            self.gpu_cache[chunk_idx]['data'][chunk_offset],
            self.gpu_cache[chunk_idx]['labels'][chunk_offset]
        )


def create_optimized_loader(h5_path, batch_size=256):
    """
    Create a DataLoader optimized for RTX 3090 GPU usage.

    Args:
        h5_path (str): Path to the HDF5 file
        batch_size (int): Batch size for training
    """
    from torch.utils.data import DataLoader

    dataset = DeepGestureDataset(h5_path)

    # Use a larger batch size since data is already on GPU
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No need for workers as data is in memory
        pin_memory=False  # Data is already on GPU
    )
    return loader
