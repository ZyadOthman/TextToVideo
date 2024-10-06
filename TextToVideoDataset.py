import os
import re
from PIL import Image
from torch.utils.data import Dataset

class TextToVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.frame_paths = []
        self.prompts = []

        for video_dir in self.video_dirs:
            frames = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.png')]
            self.frame_paths.extend(frames)
            
            with open(os.path.join(video_dir, 'prompt.txt'), 'r') as f:
                prompt = f.read().strip()

            self.prompts.extend([prompt] * len(frames))

    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path)
        prompt = self.prompts[idx]

        if self.transform:
            image = self.transform(image)

        return image, prompt
