import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import OrderedDict


class StreamingGestureDataset(Dataset):
    """
    A PyTorch Dataset optimized for systems with limited RAM but large GPU memory.
    Uses a streaming approach to efficiently move data from disk to GPU while minimizing RAM usage.
    """

    def __init__(self, h5_path, gpu_cache_size=18):
        """
        Initialize the dataset with streaming capabilities.

        Args:
            h5_path (str): Path to the HDF5 file
            gpu_cache_size (int): Size in GB to use for GPU caching (default 18GB for 3090)
        """
        self.h5_path = h5_path

        # Open file handle for reading metadata
        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_size = len(f['data'])
            # Store data shape and type for memory calculations
            self.data_shape = f['data'].shape[1:]
            self.data_dtype = f['data'].dtype

        # Calculate memory requirements
        self.sample_size_bytes = np.prod(self.data_shape) * np.dtype(self.data_dtype).itemsize

        # Calculate how many samples we can fit in GPU cache
        gpu_cache_bytes = gpu_cache_size * 1024 * 1024 * 1024  # Convert GB to bytes
        self.samples_per_cache = int(gpu_cache_bytes // self.sample_size_bytes)

        # Initialize GPU cache with OrderedDict to maintain insertion order
        self.gpu_cache = OrderedDict()

        # Calculate optimal RAM buffer size (aim for ~4GB RAM usage)
        ram_buffer_size = 4 * 1024 * 1024 * 1024  # 4GB in bytes
        self.ram_chunk_size = min(
            self.dataset_size,
            int(ram_buffer_size // self.sample_size_bytes)
        )

        self.current_ram_chunk = None
        self.current_chunk_start = None

    def _load_ram_chunk(self, start_idx):
        """
        Load a chunk of data into RAM buffer.
        """
        end_idx = min(start_idx + self.ram_chunk_size, self.dataset_size)
        with h5py.File(self.h5_path, 'r') as f:
            self.current_ram_chunk = torch.from_numpy(f['data'][start_idx:end_idx]).float()
            self.current_chunk_start = start_idx

    def _ensure_gpu_cached(self, idx):
        """
        Ensure the requested index is cached in GPU memory.
        Uses a sliding window approach to maintain the cache.
        """
        cache_block = idx // self.samples_per_cache
        cache_key = f"block_{cache_block}"

        if cache_key not in self.gpu_cache:
            # Calculate the range for this cache block
            start_idx = cache_block * self.samples_per_cache
            end_idx = min(start_idx + self.samples_per_cache, self.dataset_size)

            # Remove oldest cache block if we're at capacity (keeping 2 blocks)
            if len(self.gpu_cache) >= 2:
                self.gpu_cache.popitem(last=False)

            # Load the data through RAM buffer
            if self.current_ram_chunk is None or idx < self.current_chunk_start or \
                    idx >= self.current_chunk_start + self.ram_chunk_size:
                self._load_ram_chunk(start_idx)

            # Transfer to GPU
            block_data = self.current_ram_chunk[:end_idx - start_idx].cuda(non_blocking=True)
            self.gpu_cache[cache_key] = block_data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Get a sample from the dataset, managing both RAM and GPU caching.
        """
        # Ensure the data is in GPU cache
        self._ensure_gpu_cached(idx)

        # Calculate which cache block contains our index
        cache_block = idx // self.samples_per_cache
        cache_key = f"block_{cache_block}"

        # Calculate offset within the cache block
        cache_offset = idx % self.samples_per_cache

        # Return the data from GPU cache
        return self.gpu_cache[cache_key][cache_offset]


def create_streaming_loader(h5_path, batch_size=256):
    """
    Create a DataLoader optimized for streaming data to GPU.
    """
    from torch.utils.data import DataLoader

    dataset = StreamingGestureDataset(h5_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for controlled memory usage
        pin_memory=False  # Data is managed directly in GPU
    )
    return loader
