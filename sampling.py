import os
import logging
import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UCF101EarlyActionDataset(Dataset):
    def __init__(self, video_root: str, split_file: str, clip_len: int = 16, 
                 fraction: float = 0.25, transform: Optional[Callable] = None, 
                 model_type: str = 'resnet'):
        """
        args explained:
            video_root (str): Root folder containing class subfolders with videos
            split_file (str): Path to train/test split file
            clip_len (int): Number of frames per clip
            fraction (float): Fraction of the video to use (0.1, 0.25, 0.5, etc.)
            transform (callable): Transformations to apply to frames
            model_type (str): 'resnet' or 'transformer' to handle tensor shape
        """
        # validate inputs
        assert model_type in ['resnet', 'transformer'], "model_type must be 'resnet' or 'transformer'"
        assert 0 < fraction <= 1.0, f"fraction must be in (0, 1], got {fraction}"
        assert clip_len > 0, f"clip_len must be positive, got {clip_len}"
        
        self.video_root = video_root
        self.clip_len = clip_len
        self.fraction = fraction
        self.transform = transform
        self.model_type = model_type

        self.video_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._prepare_dataset(split_file)

    def _prepare_dataset(self, split_file: str) -> None:
        """Prepare dataset by reading split file and mapping class labels."""
        # map class names to indices
        classes = sorted([d for d in os.listdir(self.video_root) 
                         if os.path.isdir(os.path.join(self.video_root, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Read split file
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                video_rel_path = parts[0]

                if len(parts) > 1:
                    label = int(parts[1]) - 1
                else:
                    # Handle both forward and backslash separators
                    class_name = video_rel_path.replace('\\', '/').split('/')[0]
                    label = self.class_to_idx.get(class_name)
                    if label is None:
                        logger.warning(f"Class {class_name} not found in class_to_idx")
                        continue

                full_path = os.path.join(self.video_root, video_rel_path)
                if os.path.exists(full_path):
                    self.video_paths.append(full_path)
                    self.labels.append(label)
                else:
                    logger.warning(f"Video not found: {full_path}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def _read_frames(self, video_path: str) -> List[np.ndarray]:
        """Read frames according to clip_len and fraction, with safe padding."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        num_frames_to_use = max(1, int(total_frames * self.fraction))
        
        # evenly spaced frame indices within the chosen fraction of the video
        frame_indices = np.linspace(0, num_frames_to_use - 1, num=self.clip_len, dtype=int)
        
        frames = []
        last_valid_idx = None
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                last_valid_idx = frame_idx
            else:
                # found that the reported frame count can exceed the actual frame count, which leads to errors while training
                # fix = store last valid frame, and if the error occurs then treat that as the ending frame
                # then resample from this new valid range
                logger.warning(f"Failed to read frame {frame_idx} from {video_path}. "
                               f"Resampling within actual video length (0-{last_valid_idx}).")
                if last_valid_idx is not None and last_valid_idx > 0:
                    new_indices = np.linspace(0, last_valid_idx, num=self.clip_len, dtype=int)
                    frames = []
                    for new_idx in new_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                        ret2, frame2 = cap.read()
                        if ret2:
                            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                            frames.append(frame2)
                break
        
        cap.release()
        
        # Handle edge case: no frames read successfully
        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from {video_path}")
        
        # Pad with last frame if needed (fallback for very short videos)
        while len(frames) < self.clip_len:
            frames.append(frames[-1].copy())
        
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            frames = self._read_frames(video_path)
        except Exception as e:
            raise RuntimeError(f"Error reading video {video_path}: {e}")

        # transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        else:
            # Convert to tensors if no transform provided
            frames = [torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0 
                     for frame in frames]

        frames_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)

        # permute for transformer if needed
        if self.model_type == 'transformer':
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)

        return frames_tensor, label