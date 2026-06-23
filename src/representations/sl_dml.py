import numpy as np
from representations.base_representation import BaseRepresentation


class SlDMLRepresentation(BaseRepresentation):
    """
    SL-DML encoding: concatenate x, y, z channels and normalize to [0, 1].
    Input shape: (n_landmarks, n_frames).
    Output shape: (n_landmarks * 3, n_frames).
    """

    name = "SL-DML"

    def transform(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        t = np.concatenate([x, y, z], axis=1)
        t -= np.min(t)
        max_val = np.max(t)
        if max_val > 0:
            t /= max_val
        return t
