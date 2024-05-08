import torch

from . import Categorical
from .. import util
import gymnasium.spaces

OriginalDiscrete = gymnasium.spaces.Discrete #will be monkey patched later

class GymDiscrete(Categorical, OriginalDiscrete):
    def __init__(self, n, start=0, name="Gym Discrete", address_suffix='', batch_shape=torch.Size(), event_shape=torch.Size()):
        OriginalDiscrete.__init__(self, n, start=start)
        probs = torch.ones(n) / n  # Uniform distribution by default
        Categorical.__init__(self, probs=probs)

    def sample(self, mask=None):
        if mask is not None and self._torch_dist is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a numpy array."
            assert mask.shape == (self.n,), f"Mask shape must be ({self.n},), got {mask.shape}."
            # Convert mask to torch tensor and apply to probabilities
            mask_tensor = torch.from_numpy(mask).float()
            original_probs = self._torch_dist.probs
            masked_probs = original_probs * mask_tensor
            # Re-normalize the probabilities
            assert torch.sum(masked_probs) > 0, "All actions are masked. No valid sample."
            normalized_probs = masked_probs / torch.sum(masked_probs)
            self._torch_dist.probs = normalized_probs
            sample = self.start + super().sample()
            self._torch_dist.probs = original_probs
            return sample
        return self.start + super().sample()
    
    def log_prob(self, value, sum=False):
        adjusted_value = value - self.start
        return super().log_prob(adjusted_value, sum)

    def prob(self, value):
        adjusted_value = value - self.start
        return super().prob(adjusted_value)
