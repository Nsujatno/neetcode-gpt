import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max = np.max(z)
        denominator = 0
        res = []
        for num in z:
            denominator += np.exp(num - max)
        
        for num in z:
            numerator = np.exp(num - max)
            res.append(numerator / denominator)
        
        return np.round(res, 4)
