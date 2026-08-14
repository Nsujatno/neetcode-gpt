"""
Better but not as efficient solution
"""
import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max = np.max(z)
        denominator = np.sum(np.exp(z - max))
        numerator = np.exp(z - max)
        
        return np.round(numerator / denominator, 4)
