import numpy as np
from representations.base_representation import BaseRepresentation


class SkeletonDMLRepresentation(BaseRepresentation):
    """
    Skeleton-DML image encoding.
    Concatenates x and y channel blocks into a single image.
    Input shape: (n_landmarks, n_frames).
    Output shape: (n_landmarks * 2, n_frames // 3).
    """

    name = "Skeleton-DML"

    def transform(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        n = 3
        width = x.shape[1]
        if width % n != 0:
            extra = width % n
            x = x[:, : width - extra]
            y = y[:, : width - extra]

        x = np.reshape(x, (x.shape[0], -1, n))
        y = np.reshape(y, (y.shape[0], -1, n))
        return np.concatenate([x, y], axis=1)
