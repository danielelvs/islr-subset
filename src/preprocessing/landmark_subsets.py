from __future__ import annotations
"""
Landmark subset definitions.

Each function returns a list of global holistic indices (0-542):
    0   – 467  → face   (468 points)
    468 – 488  → left hand (21 points)
    489 – 521  → pose  (33 points)
    522 – 542  → right hand (21 points)
"""

import numpy as np
# import mediapipe as mp

import numpy as np

HAND_LANDMARKS = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]
POSE_LANDMARKS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# HAND_LANDMARKS = [lm.name.lower() for lm in mp.solutions.hands.HandLandmark]
# POSE_LANDMARKS = [lm.name.lower() for lm in mp.solutions.pose.PoseLandmark]


# ──────────────────────────────────────────────────────────────────────────────
# Subset functions
# ──────────────────────────────────────────────────────────────────────────────

def all_landmarks() -> list[int]:
    """All 543 MediaPipe holistic landmarks."""
    return list(range(543))


def subset_1st() -> list[int]:
    """
    1st-place Kaggle ASL subset adapted for LIBRAS.
    Includes lips, both hands, nose tip, and eye contours.
    """
    NOSE = [1, 2, 98, 327]
    LIP = [
        0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
    LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
    LHAND = list(range(468, 489))
    RHAND = list(range(522, 543))
    return LIP + LHAND + RHAND + NOSE + REYE + LEYE


def subset_2nd() -> list[int]:
    """
    2nd-place Kaggle ASL subset adapted for LIBRAS.
    Compact lip points, both hands, and upper-body pose.
    """
    lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291][::2][1:]
    lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291][::2][1:]
    lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308][::2][1:]
    lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308][::2][1:]
    LIPS = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
    POSE = [489, 490, 492, 493, 494, 498, 499, 500, 501, 502, 503, 504, 505, 506,
            507, 508, 509, 510, 511, 512]
    LHAND = list(range(468, 489))
    RHAND = list(range(522, 543))
    return LIPS + LHAND + POSE + RHAND


def subset_laines() -> list[int]:
    """David/Laines subset: selected face landmarks + hands + upper-body pose."""
    FACE = [46, 52, 53, 65, 295, 283, 282, 276, 7, 159, 155, 145, 382, 386, 249, 374, 324, 13, 78, 14]
    BODY = [489, 500, 501, 502, 503]
    # BODY = [500, 501, 502, 503, 504, 505]
    LHAND = list(range(468, 489))
    RHAND = list(range(522, 543))
    return FACE + LHAND + BODY + RHAND


def subset_arcanjo() -> list[int]:
    """Arcanjo subset: hands + full pose (no face)."""
    LHAND = list(range(468, 489))
    POSE  = list(range(489, 522))
    RHAND = list(range(522, 543))
    return LHAND + POSE + RHAND


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

SUBSETS: dict[str, callable] = {
    "all":    all_landmarks,
    "1st":    subset_1st,
    "2nd":    subset_2nd,
    "laines": subset_laines,
    "arcanjo": subset_arcanjo,
}


# ──────────────────────────────────────────────────────────────────────────────
# Column mapping (MediaPipe holistic index → CSV column names)
# ──────────────────────────────────────────────────────────────────────────────

def indices_to_columns(landmark_indices: list[int]) -> list[str]:
    """Convert a list of holistic indices to CSV column names (x, y, z triples)."""
    cols = []
    for idx in landmark_indices:
        if idx < 468:
            cols += [f"face_{idx}_x", f"face_{idx}_y", f"face_{idx}_z"]
        elif 468 <= idx < 489:
            h = idx - 468
            cols += [
                f"hand_0_{HAND_LANDMARKS[h]}_x",
                f"hand_0_{HAND_LANDMARKS[h]}_y",
                f"hand_0_{HAND_LANDMARKS[h]}_z",
            ]
        elif 489 <= idx < 522:
            p = idx - 489
            cols += [
                f"pose_{POSE_LANDMARKS[p]}_x",
                f"pose_{POSE_LANDMARKS[p]}_y",
                f"pose_{POSE_LANDMARKS[p]}_z",
            ]
        elif 522 <= idx < 543:
            h = idx - 522
            cols += [
                f"hand_1_{HAND_LANDMARKS[h]}_x",
                f"hand_1_{HAND_LANDMARKS[h]}_y",
                f"hand_1_{HAND_LANDMARKS[h]}_z",
            ]
    return cols
