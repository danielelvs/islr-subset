from base_extractor import BaseExtractor
import mediapipe as mp
import cv2
import numpy as np

HAND_POSITIONS = [landmark.name.lower() for landmark in mp.solutions.hands.HandLandmark]
POSE_POSITIONS = [landmark.name.lower() for landmark in mp.solutions.pose.PoseLandmark]

class MediaPipeExtractor(BaseExtractor):
    def __init__(self, min_detection_confidence=0.4, min_tracking_confidence=0.4):
        mp_holistic = mp.solutions.holistic
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def get_video_landmarks(self, video_path):
        video = cv2.VideoCapture(video_path)

        while video.isOpened():
            check, img = video.read()
            if check:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield self.holistic.process(imgRGB)
            else:
                break
        video.release()
    
    def extract_landmarks(self, results):
        landmarks = {}
        landmarks.update(self.extract_face(results.face_landmarks))
        landmarks.update(self.extract_hand(results.left_hand_landmarks, 0))
        landmarks.update(self.extract_pose(results.pose_landmarks))
        landmarks.update(self.extract_hand(results.right_hand_landmarks, 1))
        return landmarks
    
    def extract_hand(self, hand, hand_id):
        landmarks = {}
        if hand:
            for i, landmark in enumerate(hand.landmark):
                landmarks[f"hand_{hand_id}_{HAND_POSITIONS[i]}_x"] = landmark.x
                landmarks[f"hand_{hand_id}_{HAND_POSITIONS[i]}_y"] = landmark.y
                landmarks[f"hand_{hand_id}_{HAND_POSITIONS[i]}_z"] = landmark.z
        else:
            for pos in HAND_POSITIONS:
                landmarks[f"hand_{hand_id}_{pos}_x"] = np.nan
                landmarks[f"hand_{hand_id}_{pos}_y"] = np.nan
                landmarks[f"hand_{hand_id}_{pos}_z"] = np.nan
        return landmarks

    def extract_face(self, face):
        landmarks = {}
        if face:
            landmark_count = 0
            if type(face) == list:
                if len(face) > 1:
                    print("More than one face detected. Using first face")
                face = face[0]
            for l in face.landmark:
                landmarks[f"face_{landmark_count}_x"] = l.x
                landmarks[f"face_{landmark_count}_y"] = l.y
                landmarks[f"face_{landmark_count}_z"] = l.z
                landmark_count += 1
        else:
            for l in range(468):
                landmarks[f"face_{l}_x"] = np.nan
                landmarks[f"face_{l}_y"] = np.nan
                landmarks[f"face_{l}_z"] = np.nan
        return landmarks

    def extract_pose(self, pose):
        landmarks = {}
        if pose:
            landmark_count = 0
            for l in pose.landmark:
                landmarks[f"pose_{POSE_POSITIONS[landmark_count]}_x"] = l.x
                landmarks[f"pose_{POSE_POSITIONS[landmark_count]}_y"] = l.y
                landmarks[f"pose_{POSE_POSITIONS[landmark_count]}_z"] = l.z
                landmark_count += 1
        else:
            for l in range(len(POSE_POSITIONS)):
                landmarks[f"pose_{POSE_POSITIONS[l]}_x"] = np.nan
                landmarks[f"pose_{POSE_POSITIONS[l]}_y"] = np.nan
                landmarks[f"pose_{POSE_POSITIONS[l]}_z"] = np.nan
        return landmarks