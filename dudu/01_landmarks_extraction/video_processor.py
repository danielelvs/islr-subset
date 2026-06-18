import time
import numpy as np
import pandas as pd
from tqdm import tqdm

class DefaultVideoProcessor:
    def __init__(self, extractor):
        self.extractor = extractor
    
    def process_all(self, videos_to_process, output_path, chunk_size=1000):
        chunk_data = []
        is_first_chunk = True
        total_start_time = time.time()

        for video_path, video_name, category, signaler, index in tqdm(videos_to_process, desc="Processando vídeos"):
            for frame in self._generate_frames(video_path, category, video_name, signaler):
                chunk_data.append(frame)

                if len(chunk_data) >= chunk_size:
                    self._write_chunk(chunk_data, output_path, is_first_chunk)
                    chunk_data.clear()
                    is_first_chunk = False

        if chunk_data:
            self._write_chunk(chunk_data, output_path, is_first_chunk)
            chunk_data.clear()

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"Total execution time: {int(total_time/60)}m or {total_time}s")
    
    def _generate_frames(self, video_path, category_index, video_name, signaler):
        for i, l in enumerate(self.extractor.get_video_landmarks(video_path)):
            frame_data = {
                "category": category_index,
                "video_name": video_name,
                "frame": i,
                "person": signaler
            }
            frame_data.update(self.extractor.extract_landmarks(l))

            yield frame_data

    def _write_chunk(self, chunk_data, output_path, save_header):
        df_chunk = pd.DataFrame(chunk_data)
        
        hand_columns = [col for col in df_chunk.columns if col.startswith("hand_")]
        face_columns = [col for col in df_chunk.columns if col.startswith("face_")]
        if hand_columns: df_chunk["missing_hand"] = df_chunk[hand_columns].isna().all(axis=1)
        if face_columns: df_chunk["missing_face"] = df_chunk[face_columns].isna().all(axis=1)

        df_chunk.to_csv(output_path, mode='a', header=save_header, index=False, na_rep="NaN")


class UFOPVideoProcessor(DefaultVideoProcessor):
    def __init__(self, extractor, labels, frames_threshold=15):
        super().__init__(extractor)
        self.labels = labels
        self.frames_threshold = frames_threshold
    
    def process_all(self, videos_to_process, output_path, chunk_size=1000):
        chunk_data = []
        is_first_chunk = True
        total_start_time = time.time()

        for video_path, video_name, category, signaler, index in tqdm(videos_to_process, desc="Processando vídeos"):
            all_frames = list(self._generate_frames(video_path, category, video_name, signaler))
            if not all_frames:
                continue
            df_video = pd.DataFrame(all_frames)

            video_labels = self.labels.get(video_name)
            if not video_labels:
                continue

            for i, label in enumerate(video_labels):
                prev_label = video_labels[i-1] if i > 0 else None
                next_label = video_labels[i+1] if i < len(video_labels) - 1 else None

                start_frame = self._get_start_frame(label, prev_label)
                end_frame = self._get_end_frame(label, next_label)
                category = self._get_category(label)

                df_segment = df_video[(df_video["frame"] >= start_frame) & (df_video["frame"] <= end_frame)].copy()
                if df_segment.empty:
                    continue
                df_segment["video_name"] = f"{video_name}_{i}"
                df_segment["frame"] = np.arange(len(df_segment))
                df_segment["person"] = signaler
                df_segment["category"] = category

                chunk_data.extend(df_segment.to_dict("records"))

                if len(chunk_data) >= chunk_size:
                    self._write_chunk(chunk_data, output_path, is_first_chunk)
                    chunk_data.clear()
                    is_first_chunk = False
        
        if chunk_data:
            self._write_chunk(chunk_data, output_path, is_first_chunk)
            chunk_data.clear()

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"Total execution time: {int(total_time/60)}m or {total_time}s")       

    def _get_category(self, label):
        return int(label.split(":")[1])

    def _get_frame(self, label, index):
        return int(label.split(":")[0].split("-")[index])

    def _get_start_frame(self, label, prev_label):
        if prev_label is None:
            last_frame = 0
        else:
            last_frame = self._get_frame(prev_label, 1)
        start_frame = self._get_frame(label, 0)
        if last_frame >= start_frame - self.frames_threshold:
            return start_frame
        return start_frame - self.frames_threshold
    
    def _get_end_frame(self, label, next_label):
        if next_label is None:
            first_frame = 0
        else:
            first_frame = self._get_frame(next_label, 0)
        end_frame = self._get_frame(label, 1)
        if first_frame <= end_frame + self.frames_threshold:
            return end_frame
        return end_frame + self.frames_threshold