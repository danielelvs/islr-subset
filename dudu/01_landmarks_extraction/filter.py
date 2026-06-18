import pandas as pd
import mediapipe as mp
import numpy as np
import os

HAND_LANDMARKS = [landmark.name.lower() for landmark in mp.solutions.hands.HandLandmark]
POSE_LANDMARKS = [landmark.name.lower() for landmark in mp.solutions.pose.PoseLandmark]

def all_landmarks():
    FACE = np.arange(0, 468).tolist()       # 0-467 = 468
    LHAND = np.arange(468, 489).tolist()    # 468-488 = 21
    POSE = np.arange(489, 522).tolist()     # 489-521 = 33
    RHAND = np.arange(522, 543).tolist()    # 522-542 = 21

    return FACE + LHAND + POSE + RHAND      # Total = 543

def first_place_filter():
    NOSE=[
        1,2,98,327
    ]

    LNOSE = [98]
    RNOSE = [327]
    LIP = [ 0, 
        61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
    RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

    POSE = [500, 502, 504, 501, 503, 505, 512, 513]
    LPOSE = [513,505,503,501]
    RPOSE = [512,504,502,500]

    REYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
    ]
    LEYE = [
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
    ]

    LHAND = np.arange(468, 489).tolist()
    RHAND = np.arange(522, 543).tolist()

    return LIP + LHAND + RHAND + NOSE + REYE + LEYE

def second_place_filter():
    lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291][::2][1:]
    lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291][::2][1:]
    lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308][::2][1:]
    lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308][::2][1:]

    LIPS = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
    pose = [489, 490, 492, 493, 494, 498, 499, 500, 501, 502, 503, 504, 505, 506,
            507, 508, 509, 510, 511, 512]

    l_hand = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                481, 482, 483, 484, 485, 486, 487, 488]
    r_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

    return LIPS + l_hand + pose + r_hand

def david_filter():
    # FACE
    RIGHT_EYEBROW = [46, 52, 53, 65]
    LEFT_EYEBROW = [295, 283, 282, 276]
    RIGHT_EYE = [7, 159, 155, 145]
    LEFT_EYE = [382, 386, 249, 374]
    MOUTH = [324, 13, 78, 14]

    FACE_LANDMARKS = RIGHT_EYEBROW + LEFT_EYEBROW + RIGHT_EYE + LEFT_EYE + MOUTH

    # BODY
    NOSE = [489]
    RIGHT_SHOULDER = [500]
    LEFT_SHOULDER = [501]
    RIGHT_ELBOW = [502]
    LEFT_ELBOW = [503]
    # MIDDLE_CHEST # MIDPOINT BETWEEN THE SHOULDERS

    BODY_LANDMARKS = NOSE + RIGHT_SHOULDER + LEFT_SHOULDER + RIGHT_ELBOW + LEFT_ELBOW

    # HANDS
    LHAND = np.arange(468, 489).tolist()
    RHAND = np.arange(522, 543).tolist()

    POINT_LANDMARKS = FACE_LANDMARKS + LHAND + BODY_LANDMARKS + RHAND
    return POINT_LANDMARKS

def arcanjo_filter():
    LHAND = np.arange(468, 489).tolist()
    POSE = np.arange(489, 522).tolist()
    RHAND = np.arange(522, 543).tolist()

    return LHAND + POSE + RHAND

def map_holistic_indices_to_columns(landmark_indices):
    columns_to_keep = []
    for idx in landmark_indices:
        if idx < 468:
            columns_to_keep.extend([f"face_{idx}_x", f"face_{idx}_y", f"face_{idx}_z"])
        elif 468 <= idx < 489:
            hand_idx = idx - 468
            columns_to_keep.extend([f"hand_0_{HAND_LANDMARKS[hand_idx]}_x", f"hand_0_{HAND_LANDMARKS[hand_idx]}_y", f"hand_0_{HAND_LANDMARKS[hand_idx]}_z"])
        elif 489 <= idx < 522:
            pose_idx = idx - 489
            columns_to_keep.extend([f"pose_{POSE_LANDMARKS[pose_idx]}_x", f"pose_{POSE_LANDMARKS[pose_idx]}_y", f"pose_{POSE_LANDMARKS[pose_idx]}_z"])
        elif 522 <= idx < 543:
            hand_idx = idx - 522
            columns_to_keep.extend([f"hand_1_{HAND_LANDMARKS[hand_idx]}_x", f"hand_1_{HAND_LANDMARKS[hand_idx]}_y", f"hand_1_{HAND_LANDMARKS[hand_idx]}_z"])

    return columns_to_keep

def write(df, landmark_columns, metadata_columns, final_path, filter_name):
    df[landmark_columns] = df.groupby("video_name")[landmark_columns].transform(interpolate)   

    df = df.fillna(0)

    final_columns = metadata_columns + landmark_columns
    cols_in_chunk = [col for col in final_columns if col in df.columns]

    if len(cols_in_chunk) != len(final_columns):
        raise ValueError("Some columns are missing in the chunk.")
    
    filtered_chunk = df[cols_in_chunk].copy()

    if filter_name == "david":
        right_shoulder_cols = map_holistic_indices_to_columns([500])
        left_shoulder_cols = map_holistic_indices_to_columns([501])

        for right, left in zip(right_shoulder_cols, left_shoulder_cols):
            middle_chest_col = f"pose_middle_chest_{left[-1]}"
            filtered_chunk[middle_chest_col] = (filtered_chunk[right] + filtered_chunk[left]) / 2
    
    filtered_chunk.to_csv(final_path, index=False)

def interpolate(series):
    if series.notna().sum() < 4:
        return series.interpolate(method='linear', limit=5, limit_direction='both')
    return series.interpolate(method="cubic", limit=5, limit_direction="both")

def main():
    metadata_columns = ["category", "video_name", "frame", "person", "missing_hand", "missing_face"]
    DATASETS = ["minds", "ufop"]
    FILTERS = {
        "1": first_place_filter,
        "2": second_place_filter,
        "all": all_landmarks,
        "david": david_filter,
        "arcanjo": arcanjo_filter,
    }

    for dataset in DATASETS:
        original_path = f"../00_datasets/dataset_output/{dataset}/{dataset}_mediapipe.csv"
        for filter_name, filter_func in FILTERS.items():
            final_path = f"../00_datasets/dataset_output/{dataset}/{dataset}_mediapipe_filtered-{filter_name}.csv"
            if os.path.exists(final_path):
                os.remove(final_path)
                print(f"Removed existing file: {final_path}")
            point_landmarks = filter_func()
            landmark_columns = map_holistic_indices_to_columns(point_landmarks)

            df = pd.read_csv(original_path, na_values=["NaN"])
            write(df, landmark_columns, metadata_columns, final_path, filter_name)

if __name__ == "__main__":
    main()