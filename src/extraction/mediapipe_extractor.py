from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

from extraction.base_extractor import BaseExtractor

HAND_POSITIONS = [landmark.name.lower() for landmark in mp.solutions.hands.HandLandmark]
POSE_POSITIONS = [landmark.name.lower() for landmark in mp.solutions.pose.PoseLandmark]


class MediaPipeExtractor(BaseExtractor):
    def __init__(self, min_detection_confidence: float = 0.4, min_tracking_confidence: float = 0.4):
        mp_holistic = mp.solutions.holistic
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def get_video_landmarks(self, video_path: str):
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            check, img = video.read()
            if check:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield self.holistic.process(img_rgb)
            else:
                break
        video.release()

    def extract_landmarks(self, results) -> dict:
        landmarks = {}
        landmarks.update(self._extract_face(results.face_landmarks))
        landmarks.update(self._extract_hand(results.left_hand_landmarks, hand_id=0))
        landmarks.update(self._extract_pose(results.pose_landmarks))
        landmarks.update(self._extract_hand(results.right_hand_landmarks, hand_id=1))
        return landmarks

    def _extract_hand(self, hand, hand_id: int) -> dict:
        landmarks = {}
        if hand:
            for i, lm in enumerate(hand.landmark):
                landmarks[f"hand_{hand_id}_{HAND_POSITIONS[i]}_x"] = lm.x
                landmarks[f"hand_{hand_id}_{HAND_POSITIONS[i]}_y"] = lm.y
                landmarks[f"hand_{hand_id}_{HAND_POSITIONS[i]}_z"] = lm.z
        else:
            for pos in HAND_POSITIONS:
                landmarks[f"hand_{hand_id}_{pos}_x"] = np.nan
                landmarks[f"hand_{hand_id}_{pos}_y"] = np.nan
                landmarks[f"hand_{hand_id}_{pos}_z"] = np.nan
        return landmarks

    def _extract_face(self, face) -> dict:
        landmarks = {}
        if face:
            if isinstance(face, list):
                if len(face) > 1:
                    print("More than one face detected. Using first face.")
                face = face[0]
            for i, lm in enumerate(face.landmark):
                landmarks[f"face_{i}_x"] = lm.x
                landmarks[f"face_{i}_y"] = lm.y
                landmarks[f"face_{i}_z"] = lm.z
        else:
            for i in range(468):
                landmarks[f"face_{i}_x"] = np.nan
                landmarks[f"face_{i}_y"] = np.nan
                landmarks[f"face_{i}_z"] = np.nan
        return landmarks

    def _extract_pose(self, pose) -> dict:
        landmarks = {}
        if pose:
            for i, lm in enumerate(pose.landmark):
                landmarks[f"pose_{POSE_POSITIONS[i]}_x"] = lm.x
                landmarks[f"pose_{POSE_POSITIONS[i]}_y"] = lm.y
                landmarks[f"pose_{POSE_POSITIONS[i]}_z"] = lm.z
        else:
            for pos in POSE_POSITIONS:
                landmarks[f"pose_{pos}_x"] = np.nan
                landmarks[f"pose_{pos}_y"] = np.nan
                landmarks[f"pose_{pos}_z"] = np.nan
        return landmarks
