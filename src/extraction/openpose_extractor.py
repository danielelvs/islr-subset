import os
import sys
import cv2
import numpy as np

from extraction.base_extractor import BaseExtractor

_OPENPOSE_DIR = os.path.join(os.path.dirname(__file__), "bin", "openpose", "python")

try:
    sys.path.append(os.path.join(_OPENPOSE_DIR, "../bin/python/openpose/Release"))
    os.environ["PATH"] = (
        os.environ["PATH"]
        + ";"
        + os.path.join(_OPENPOSE_DIR, "../x64/Release")
        + ";"
        + os.path.join(_OPENPOSE_DIR, "../bin")
    )
    import pyopenpose as op
except ImportError as e:
    raise ImportError(
        "OpenPose library not found. Enable `BUILD_PYTHON` in CMake and place this "
        "script in the correct folder."
    ) from e


class OpenPoseExtractor(BaseExtractor):
    def __init__(self, max_num_hands: int = 2):
        self.max_num_hands = max_num_hands
        params = {
            "model_folder": r"bin\openpose\models/",
            "face": True,
            "hand": True,
        }
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

    def get_video_landmarks(self, video_path: str):
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            check, img = video.read()
            if check:
                yield self._extract_image(img)
            else:
                break
        video.release()

    def extract_landmarks(self, results: dict) -> dict:
        landmarks = {}
        landmarks.update(self._extract_face(results["face_landmarks"]))
        landmarks.update(self._extract_hands(results["hands_landmarks"]))
        landmarks.update(self._extract_pose(results["pose_landmarks"]))
        return landmarks

    def _extract_image(self, image) -> dict:
        datum = op.Datum()
        datum.cvInputData = image
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
        return {
            "hands_landmarks": self._normalize_hand(datum.handKeypoints, image),
            "face_landmarks": self._normalize_face(datum.faceKeypoints, image),
            "pose_landmarks": self._normalize_pose(datum.poseKeypoints, image),
        }

    def _extract_hands(self, hand) -> dict:
        landmarks = {}
        hand_count = 0
        if hand is not None:
            for h in hand:
                if hand_count >= self.max_num_hands:
                    break
                for j, point in enumerate(h[0]):
                    landmarks[f"hand_{hand_count}_{j}_x"] = point[0]
                    landmarks[f"hand_{hand_count}_{j}_y"] = point[1]
                    landmarks[f"hand_{hand_count}_{j}_z"] = point[2]
                hand_count += 1
        for h in range(hand_count, self.max_num_hands):
            for j in range(21):
                landmarks[f"hand_{h}_{j}_x"] = np.nan
                landmarks[f"hand_{h}_{j}_y"] = np.nan
                landmarks[f"hand_{h}_{j}_z"] = np.nan
        return landmarks

    def _extract_face(self, face) -> dict:
        landmarks = {}
        if face is not None:
            if isinstance(face, (list, np.ndarray)) and len(face) > 1:
                print("More than one face detected. Using first face.")
            face = face[0]
            for i, point in enumerate(face):
                landmarks[f"face_{i}_x"] = point[0]
                landmarks[f"face_{i}_y"] = point[1]
                landmarks[f"face_{i}_z"] = point[2]
        else:
            for i in range(70):
                landmarks[f"face_{i}_x"] = np.nan
                landmarks[f"face_{i}_y"] = np.nan
                landmarks[f"face_{i}_z"] = np.nan
        return landmarks

    def _extract_pose(self, pose) -> dict:
        landmarks = {}
        if pose is not None:
            for i, point in enumerate(pose[0]):
                landmarks[f"pose_{i}_x"] = point[0]
                landmarks[f"pose_{i}_y"] = point[1]
                landmarks[f"pose_{i}_z"] = point[2]
        else:
            for i in range(25):
                landmarks[f"pose_{i}_x"] = np.nan
                landmarks[f"pose_{i}_y"] = np.nan
                landmarks[f"pose_{i}_z"] = np.nan
        return landmarks

    @staticmethod
    def _normalize_hand(hand, image):
        for h in hand:
            h[0][:, 0] /= image.shape[1]
            h[0][:, 1] /= image.shape[0]
        return hand

    @staticmethod
    def _normalize_face(face, image):
        face[0][:, 0] /= image.shape[1]
        face[0][:, 1] /= image.shape[0]
        return face

    @staticmethod
    def _normalize_pose(pose, image):
        pose[0][:, 0] /= image.shape[1]
        pose[0][:, 1] /= image.shape[0]
        return pose
