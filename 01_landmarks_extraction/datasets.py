from abc import ABC, abstractmethod
import os

class BaseDataset(ABC):
    path_key = ""
    
    def __init__(self, base_path):
        self.dataset_path = os.path.join(base_path, self.path_key)
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset dir {self.dataset_path} does not exist.")
    
    @abstractmethod
    def prepare_data(self):
        pass

    def get_processor(self, extractor):
        from video_processor import DefaultVideoProcessor
        return DefaultVideoProcessor(extractor)
    
    @staticmethod
    def create(dataset_name, base_path):
        if dataset_name == "ufop":
            return UFOPDataset(base_path)
        elif dataset_name == "vlibras":
            return VLibrasDataset(base_path)
        elif dataset_name == "minds":
            return MINDSDataset(base_path)
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Use 'ufop', 'vlibras', 'minds'.")

class UFOPDataset(BaseDataset):
    path_key = "LIBRAS-UFOP"

    def __init__(self, base_path):
        super().__init__(base_path)
        self.labels = self._load_labels()
        self.frames_threshold = 15
    
    def prepare_data(self):
        videos_to_process = []

        for folder_name in os.listdir(self.dataset_path):
            if not folder_name.startswith("p"):
                continue

            splitted = folder_name.split("_")

            subject_id = splitted[0][1:]
            video_name = folder_name # use folder name as video name to avoid repeated Color.avi
            category = -1 # placeholder for future processing (for sign) TODO: rename category to sign
            video_path = os.path.join(self.dataset_path, folder_name, "Color.avi")

            videos_to_process.append((video_path, video_name, category, subject_id, subject_id))

        return videos_to_process
    
    def get_processor(self, extractor):
        from video_processor import UFOPVideoProcessor
        return UFOPVideoProcessor(extractor, self.labels, frames_threshold=self.frames_threshold)
    
    def _load_labels(self):
        labels_path = os.path.join(self.dataset_path, "labels.txt")
        if not os.path.exists(labels_path):
            raise ValueError(f"Labels file {labels_path} does not exist.")
        
        labels = {}
        with open(labels_path, "r") as f:
            for line in f:
                split = line.strip().split(" ")
                label_name = split[0]
                label_values = split[1:]
                labels[label_name] = label_values
        return labels

class VLibrasDataset(BaseDataset):
    path_key = "v-librasil/videos UFPE (V-LIBRASIL)/data"

    def prepare_data(self):
        videos_to_process = []
        
        for video in os.listdir(self.dataset_path):
            video_path = os.path.join(self.dataset_path, video)
            video_name = video
            video_sign = video.split("_")[0]
            signaler = video[-5]

            videos_to_process.append((video_path, video_name, video_sign, signaler, signaler))

        return videos_to_process

class MINDSDataset(BaseDataset):
    path_key = "MINDS"

    def prepare_data(self):
        videos_to_process = []

        signalers = os.listdir(self.dataset_path)
        for signaler in signalers:
            if not os.path.isdir(os.path.join(self.dataset_path, signaler)):
                continue
            signaler_dir = os.path.join(self.dataset_path, signaler, "Canon")
            videos = os.listdir(signaler_dir)
            for video in videos:
                if not video.endswith(".mp4"):
                    continue
                video_path = os.path.join(signaler_dir, video)
                video_name = video
                video_sign = video.split("Sinalizador")[0][2:]
                signaler_id = int(signaler[-2:])
                
                videos_to_process.append((video_path, video_name, video_sign, signaler_id, signaler_id))

        return videos_to_process