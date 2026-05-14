import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        sum=0
        for i in range(len(y_true)):
            y_p=np.clip(y_pred[i], 1e-7, 1 - 1e-7)
            sum=sum+(y_true[i]*np.log(y_p))+(1-y_true[i])*np.log(1-y_p)
        sum=-sum/len(y_true)
        return round(float(sum),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0

        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        for i in range(len(y_true)):
            loss += np.sum(y_true[i] * np.log(y_pred[i]))

        loss = -loss / len(y_true)

        return round(float(loss), 4)