import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxi=max(z)
        sum=0
        for i in range(len(z)):
            sum=sum+np.exp(z[i]-maxi)
        for i in range(len(z)):
            z[i]=round(np.exp(z[i]-maxi)/sum,4)
        return z
