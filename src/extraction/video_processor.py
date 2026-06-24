from __future__ import annotations
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


class DefaultVideoProcessor:
    def __init__(self, extractor):
        self.extractor = extractor

    def process_all(self, videos_to_process, output_path: str, chunk_size: int = 1000):
        chunk_data = []
        is_first_chunk = True
        total_start = time.time()

        for video_path, video_name, category, signaler, _ in tqdm(videos_to_process, desc="Processing videos"):
            for frame in self._generate_frames(video_path, category, video_name, signaler):
                chunk_data.append(frame)
                if len(chunk_data) >= chunk_size:
                    self._write_chunk(chunk_data, output_path, is_first_chunk)
                    chunk_data.clear()
                    is_first_chunk = False

        if chunk_data:
            self._write_chunk(chunk_data, output_path, is_first_chunk)

        elapsed = time.time() - total_start
        print(f"Total time: {int(elapsed / 60)}m ({elapsed:.1f}s)")

    def _generate_frames(self, video_path, category, video_name, signaler):
        for i, results in enumerate(self.extractor.get_video_landmarks(video_path)):
            frame = {
                "category": category,
                "video_name": video_name,
                "frame": i,
                "person": signaler,
            }
            frame.update(self.extractor.extract_landmarks(results))
            yield frame

    def _write_chunk(self, chunk_data, output_path, save_header: bool):
        df = pd.DataFrame(chunk_data)
        hand_cols = [c for c in df.columns if c.startswith("hand_")]
        face_cols = [c for c in df.columns if c.startswith("face_")]
        if hand_cols:
            df["missing_hand"] = df[hand_cols].isna().all(axis=1)
        if face_cols:
            df["missing_face"] = df[face_cols].isna().all(axis=1)
        df.to_csv(output_path, mode="a", header=save_header, index=False, na_rep="NaN")


class UFOPVideoProcessor(DefaultVideoProcessor):
    def __init__(self, extractor, labels: dict, frames_threshold: int = 15):
        super().__init__(extractor)
        self.labels = labels
        self.frames_threshold = frames_threshold

    def process_all(self, videos_to_process, output_path: str, chunk_size: int = 1000):
        chunk_data = []
        is_first_chunk = True
        total_start = time.time()

        for video_path, video_name, _, signaler, _ in tqdm(videos_to_process, desc="Processing videos"):
            all_frames = list(self._generate_frames(video_path, None, video_name, signaler))
            if not all_frames:
                continue
            df_video = pd.DataFrame(all_frames)
            video_labels = self.labels.get(video_name)
            if not video_labels:
                continue

            for i, label in enumerate(video_labels):
                prev_label = video_labels[i - 1] if i > 0 else None
                next_label = video_labels[i + 1] if i < len(video_labels) - 1 else None
                start_frame = self._get_start_frame(label, prev_label)
                end_frame = self._get_end_frame(label, next_label)
                category = self._get_category(label)

                segment = df_video[
                    (df_video["frame"] >= start_frame) & (df_video["frame"] <= end_frame)
                ].copy()
                if segment.empty:
                    continue
                segment["video_name"] = f"{video_name}_{i}"
                segment["frame"] = np.arange(len(segment))
                segment["person"] = signaler
                segment["category"] = category
                chunk_data.extend(segment.to_dict("records"))

                if len(chunk_data) >= chunk_size:
                    self._write_chunk(chunk_data, output_path, is_first_chunk)
                    chunk_data.clear()
                    is_first_chunk = False

        if chunk_data:
            self._write_chunk(chunk_data, output_path, is_first_chunk)

        elapsed = time.time() - total_start
        print(f"Total time: {int(elapsed / 60)}m ({elapsed:.1f}s)")

    def _get_category(self, label: str) -> int:
        return int(label.split(":")[1])

    def _get_frame(self, label: str, index: int) -> int:
        return int(label.split(":")[0].split("-")[index])

    def _get_start_frame(self, label: str, prev_label) -> int:
        last_frame = self._get_frame(prev_label, 1) if prev_label else 0
        start = self._get_frame(label, 0)
        return start if last_frame >= start - self.frames_threshold else start - self.frames_threshold

    def _get_end_frame(self, label: str, next_label) -> int:
        first_next = self._get_frame(next_label, 0) if next_label else 0
        end = self._get_frame(label, 1)
        return end if first_next <= end + self.frames_threshold else end + self.frames_threshold
