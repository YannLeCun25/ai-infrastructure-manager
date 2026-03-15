import numpy as np

class ResourceOptimizer:
    """Optimization algorithms for GPU memory and compute efficiency."""
    def optimize_batch_size(self, gpu_mem_gb: float, model_params_million: int) -> int:
        """Heuristic-based batch size optimization."""
        base_size = (gpu_mem_gb * 1024) / (model_params_million * 4)
        return int(2 ** np.floor(np.log2(base_size)))
