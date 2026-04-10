import cv2
import argparse
from base_extractor import BaseExtractor
from datasets import BaseDataset
from tqdm import tqdm
from time import time

def get_representative_videos(videos, n=5):
    sorted_videos = sort_videos_by_duration(videos)
    if len(sorted_videos) <= n:
        print("Número de vídeos menor ou igual ao número solicitado. Retornando todos os vídeos...")
        return sorted_videos
    shortest = sorted_videos[:n]
    longest = sorted_videos[-n:]
    median_index = len(sorted_videos) // 2
    median = sorted_videos[median_index - n//2 : median_index + n//2 + n%2]
    return shortest + median + longest

def sort_videos_by_duration(videos):
    return sorted(videos, key=lambda x: (i:=get_video_info(x[0]))[0]/i[1])

def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return frames, fps

parser = argparse.ArgumentParser(description="Argumentos para a medição de latência entre MediaPipe/OpenPose")
parser.add_argument("-d", "--dataset", required=True, choices=['ufop', 'minds'], help="Nome do dataset a ser processado")
parser.add_argument("-e", "--extractor", default='mediapipe', choices=['mediapipe', 'openpose'], help="Extrator de landmarks a ser utilizado.")
parser.add_argument("-i", "--input_dir", default='../00_datasets', help="Diretório base onde os datasets brutos estão localizados.")

args = parser.parse_args()

dataset_instance = BaseDataset.create(dataset_name=args.dataset, base_path=args.input_dir)
extractor = BaseExtractor.create(args.extractor)

videos = dataset_instance.prepare_data()

representative_videos = get_representative_videos(videos, n=5)

total_time = 0
total_frames = 0

for video_path, video_name, category, signaler, index in tqdm(representative_videos):
    frames, fps_video = get_video_info(video_path)
    start_time = time()
    for _ in extractor.get_video_landmarks(video_path):
        pass # Apenas para perder tempo...
    end_time = time()
    total_time += end_time - start_time
    total_frames += frames
avg_time = total_time / len(representative_videos)
fps = total_frames / total_time

print(f"Average time per video: {avg_time:.4f}s")
print(f"Overall FPS: {fps:.4f}")
