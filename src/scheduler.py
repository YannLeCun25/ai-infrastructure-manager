import numpy as np

class GPUScheduler:
    """Intelligent GPU resource allocator for ML clusters."""
    def __init__(self, num_gpus: int):
        self.gpus = np.zeros(num_gpus) # 0 = Idle, 1 = Busy

    def allocate(self, model_id: str):
        """Finds and allocates the next available GPU."""
        idle_indices = np.where(self.gpus == 0)[0]
        if len(idle_indices) > 0:
            idx = idle_indices[0]
            self.gpus[idx] = 1
            return f"Model {model_id} allocated to GPU {idx}"
        return "No GPUs available."
