import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z1 = z - max(z)
        z2 = np.exp(z1)
        z3 = z2 / sum(z2)
        return np.round(z3, 4)
