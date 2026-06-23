import numpy as np
from representations.base_representation import BaseRepresentation


class SkeletonMagnitudeRepresentation(BaseRepresentation):
    """
    Skeleton-Magnitude image encoding.

    For each temporal scale t, computes the frame-to-frame displacement
    magnitude at each joint over that distance, producing one channel.
    Channels are stacked along axis=2.

    Input shape:  (n_landmarks, n_frames)
    Output shape: (n_landmarks, n_frames, len(temporal_scales))

    Bugs fixed vs. original:
        - diff_y and diff_z now subtract shifted_y / shifted_z (not shifted_x).
        - np.roll uses axis=1 so it rolls along the temporal axis, not the
          flattened array.
    """

    name = "Skeleton-Magnitude"

    def __init__(self, temporal_scales: list[int] | None = None):
        self.temporal_scales = temporal_scales if temporal_scales is not None else [5, 10, 15]

    def transform(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        channels = []
        for t in self.temporal_scales:
            diff = np.array(self._temporal_difference(x, y, z, t))
            channels.append(self._magnitude(diff))
        img = np.array(channels)          # (n_scales, n_landmarks, n_frames)
        return np.moveaxis(img, 0, 2)     # (n_landmarks, n_frames, n_scales)

    def _temporal_difference(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, dist: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-joint displacement over `dist` frames (rolls along temporal axis=1)."""
        # FIX: axis=1 ensures we roll along frames, not the flattened array
        sx = np.roll(x, -dist, axis=1)
        sy = np.roll(y, -dist, axis=1)
        sz = np.roll(z, -dist, axis=1)

        # Zero out wrapped-around tail so the last `dist` frames don't
        # produce spurious differences from the beginning of the video
        sx[:, -dist:] = 0
        sy[:, -dist:] = 0
        sz[:, -dist:] = 0

        # FIX: diff_y subtracts sy (not sx), diff_z subtracts sz (not sx)
        diff_x = x - sx
        diff_y = y - sy
        diff_z = z - sz

        return diff_x, diff_y, diff_z

    @staticmethod
    def _magnitude(diff: np.ndarray) -> np.ndarray:
        """Euclidean magnitude, clipped to [0, 1]."""
        mag = (diff ** 2).sum(axis=0) ** 0.5
        mag = np.clip(mag, 0.0, 1.0)
        return mag
