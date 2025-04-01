import threading
import torch
import torch.distributed as dist

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=float("inf")):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # Local statistics
        self.local_sum = torch.zeros(size, dtype=torch.float32)
        self.local_sumsq = torch.zeros(size, dtype=torch.float32)
        self.local_count = torch.zeros(1, dtype=torch.float32)

        # Global statistics
        self.total_sum = torch.zeros(size, dtype=torch.float32)
        self.total_sumsq = torch.zeros(size, dtype=torch.float32)
        self.total_count = torch.ones(1, dtype=torch.float32)  # Start with 1 to avoid division errors

        # Computed mean and std
        self.mean = torch.zeros(size, dtype=torch.float32)
        self.std = torch.ones(size, dtype=torch.float32)

        # Thread lock
        self.lock = threading.Lock()

    def update(self, v):
        v = torch.tensor(v, dtype=torch.float32).reshape(-1, self.size)
        with self.lock:
            self.local_sum += v.sum(dim=0)
            self.local_sumsq += (v ** 2).sum(dim=0)
            self.local_count += v.shape[0]

    def sync(self):
        if dist.is_initialized():  # Only sync if using distributed computing
            self.local_sum = self._torch_all_reduce(self.local_sum)
            self.local_sumsq = self._torch_all_reduce(self.local_sumsq)
            self.local_count = self._torch_all_reduce(self.local_count)

        return self.local_sum, self.local_sumsq, self.local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.clone()
            local_sum = self.local_sum.clone()
            local_sumsq = self.local_sumsq.clone()

            # Reset local values
            self.local_count.zero_()
            self.local_sum.zero_()
            self.local_sumsq.zero_()

        # Synchronize across processes (if distributed)
        sync_sum, sync_sumsq, sync_count = self.sync()

        # Update global statistics
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count

        # Compute mean and standard deviation
        self.mean = self.total_sum / self.total_count
        self.std = torch.sqrt(torch.maximum(torch.tensor(self.eps ** 2), (self.total_sumsq / self.total_count) - (self.mean ** 2)))

    def _torch_all_reduce(self, x):
        """Performs an all-reduce sum operation and averages over all processes."""
        if dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        return x

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        v = torch.tensor(v, dtype=torch.float32)
        return torch.clamp((v - self.mean) / self.std, -clip_range, clip_range)

