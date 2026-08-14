"""
Better, most efficient solution
"""
import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_max = np.max(z)
        exp_shifted = np.exp(z - z_max)
        return np.round(exp_shifted / np.sum(exp_shifted), 4)
