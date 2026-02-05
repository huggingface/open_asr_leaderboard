"""Benchmark utilities for accurate ASR model timing.

This module provides CUDA synchronization for accurate GPU timing.

Usage:
    from normalizer import cuda_sync

    # For timing
    cuda_sync(device)
    start = time.time()
    # ... model inference ...
    cuda_sync(device)
    elapsed = time.time() - start
"""

from typing import Optional, Union

# Lazy import torch to avoid import errors when torch is not installed
_torch = None


def _get_torch():
    """Lazily import torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def cuda_sync(device: Optional[Union[int, str]] = None) -> None:
    """Synchronize CUDA device for accurate timing.
    
    CUDA operations are asynchronous - without synchronization,
    time.time() measures kernel launch latency, not execution time.
    
    Args:
        device: Device index (int), device string ("cuda:0"), or None.
                If None or "cpu" or -1, no synchronization is performed.
    """
    torch = _get_torch()
    
    # Skip sync for CPU
    if device is None:
        return
    if isinstance(device, str) and "cpu" in device.lower():
        return
    if isinstance(device, int) and device < 0:
        return
    
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)
