from __future__ import annotations

"""Skeleton-magnitude image representation."""

import numpy as np

from representations.base_representation import BaseRepresentation


class SkeletonMagnitudeRepresentation(BaseRepresentation):
    """Encode multi-scale temporal displacement magnitudes as image channels.

    Input shape: ``(n_landmarks, n_frames)``.
    Output shape: ``(n_landmarks, n_frames, n_temporal_scales)``.
    """

    name = "Skeleton-Magnitude"

    def __init__(self, temporal_scales: list[int] | None = None):
        self.temporal_scales = temporal_scales or [5, 10, 15]

    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        channels = []
        for distance in self.temporal_scales:
            difference = np.asarray(
                self._temporal_difference(x, y, z, distance)
            )
            channels.append(self._magnitude(difference))
        return np.moveaxis(np.asarray(channels), 0, 2)

    @staticmethod
    def _temporal_difference(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        distance: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-landmark displacement over a temporal distance."""

        shifted_x = np.roll(x, -distance, axis=1)
        shifted_y = np.roll(y, -distance, axis=1)
        shifted_z = np.roll(z, -distance, axis=1)

        shifted_x[:, -distance:] = 0
        shifted_y[:, -distance:] = 0
        shifted_z[:, -distance:] = 0

        return x - shifted_x, y - shifted_y, z - shifted_z

    @staticmethod
    def _magnitude(difference: np.ndarray) -> np.ndarray:
        """Return the Euclidean magnitude clipped to the image range."""

        magnitude = np.sqrt(np.square(difference).sum(axis=0))
        return np.clip(magnitude, 0.0, 1.0)
