from base_extractor import BaseExtractor
import cv2
import numpy as np
import os
import sys

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = r"bin/openpose/python"
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/../bin/python/openpose/Release');
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' +  dir_path + '/../bin;'
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

class OpenPoseExtractor(BaseExtractor):
    def __init__(self, max_num_hands=2):
        self.max_num_hands = max_num_hands
        params = dict()
        params[
            "model_folder"] = r"bin\openpose\models/"
        params["face"] = True
        params["hand"] = True
        # params["net_resolution"] = "-1x256"
        # params["face_net_resolution"] = "256x256"
        # params["hand_net_resolution"] = "256x256"

        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()
    
    def get_video_landmarks(self, video_path):
        video = cv2.VideoCapture(video_path)
        
        while video.isOpened():
            check, img = video.read()
            if check:
                landmarks = self._extract_image(img)
                yield landmarks
            else:
                break
        
        video.release()
    
    def extract_landmarks(self, results):
        landmarks = {}
        landmarks.update(self.extract_face(results["face_landmarks"]))
        landmarks.update(self.extract_hands(results["hands_landmarks"]))
        landmarks.update(self.extract_pose(results["pose_landmarks"]))
        return landmarks

    def extract_hands(self, hand):
        landmarks = {}
        hand_count = 0
        if hand is not None:
            for h in hand:
                if hand_count >= self.max_num_hands:
                    break
                landmark_count = 0
                for l in h[0]:
                    landmarks[f"hand_{hand_count}_{landmark_count}_x"] = l[0]
                    landmarks[f"hand_{hand_count}_{landmark_count}_y"] = l[1]
                    landmarks[f"hand_{hand_count}_{landmark_count}_z"] = l[2]
                    landmark_count += 1
                hand_count += 1
        for h in range(hand_count, self.max_num_hands):
            for l in range(21):
                landmarks[f"hand_{h}_{l}_x"] = np.nan
                landmarks[f"hand_{h}_{l}_y"] = np.nan
                landmarks[f"hand_{h}_{l}_z"] = np.nan
        return landmarks

    def extract_face(self, face):
        landmarks = {}
        if face is not None:
            landmark_count = 0
            if type(face) in [list, np.ndarray]:
                if len(face) > 1:
                    # raise "Não era para ser"
                    print("More than one face detected. Using first face")
                face = face[0]
            for l in face:
                landmarks[f"face_{landmark_count}_x"] = l[0]
                landmarks[f"face_{landmark_count}_y"] = l[1]
                landmarks[f"face_{landmark_count}_z"] = l[2]
                landmark_count += 1
        else:
            for l in range(70):
                landmarks[f"face_{l}_x"] = np.nan
                landmarks[f"face_{l}_y"] = np.nan
                landmarks[f"face_{l}_z"] = np.nan
        return landmarks

    def extract_pose(self, pose):
        landmarks = {}
        if pose is not None:
            landmark_count = 0
            for l in pose[0]:
                landmarks[f"pose_{landmark_count}_x"] = l[0]
                landmarks[f"pose_{landmark_count}_y"] = l[1]
                landmarks[f"pose_{landmark_count}_z"] = l[2]
                landmark_count += 1
        else:
            for l in range(25):
                landmarks[f"pose_{l}_x"] = np.nan
                landmarks[f"pose_{l}_y"] = np.nan
                landmarks[f"pose_{l}_z"] = np.nan
        return landmarks
    
    def _extract_image(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

        landmarks = {
            "hands_landmarks": self.normalize_hand(datum.handKeypoints, image),
            "face_landmarks": self.normalize_face(datum.faceKeypoints, image),
            "pose_landmarks": self.normalize_pose(datum.poseKeypoints, image)
        }

        return landmarks

    @staticmethod
    def normalize_hand(hand, image):
        for h in hand:
            h[0][:, 0] = h[0][:, 0] / image.shape[1]
            h[0][:, 1] = h[0][:, 1] / image.shape[0]
        return hand

    @staticmethod
    def normalize_face(face, image):
        face[0][:, 0] = face[0][:, 0] / image.shape[1]
        face[0][:, 1] = face[0][:, 1] / image.shape[0]
        return face

    @staticmethod
    def normalize_pose(pose, image):
        pose[0][:, 0] = pose[0][:, 0] / image.shape[1]
        pose[0][:, 1] = pose[0][:, 1] / image.shape[0]
        return pose
