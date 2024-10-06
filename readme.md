# TextToVideo - Generating Videos from Text Descriptions with GANs

## Overview

**TextToVideo** is a deep learning project that generates videos based on textual prompts using a Generative Adversarial Network (GAN) architecture. The system learns to associate specific movement patterns of shapes (like circles) with text descriptions and generates corresponding video sequences. The project creates a dataset of synthetic videos, where a circle follows various movement patterns, and then trains a GAN to generate similar videos conditioned on text prompts.

## Features

- **Video Dataset Creation**: Generates a large dataset of videos featuring simple shapes (like circles) with predefined movement patterns (e.g., moving left, diagonally, bouncing).
- **Text-to-Video Generation**: Uses GANs (Generator and Discriminator models) to generate video sequences conditioned on textual descriptions.
- **Shape and Movement Variations**: Includes a variety of shapes and movement patterns for diverse training data.
- **Custom Dataset Loader**: A custom `TextToVideoDataset` class for loading the generated video data and corresponding text prompts.
- **Training and Evaluation with GANs**: Implements a training loop using PyTorch for GAN-based text-to-video generation.

## Example Usage

### Step 1: Video Dataset Creation

The script generates a dataset of 1,000 synthetic videos where a circle moves in different directions (e.g., left, right, diagonally).

```python
import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw

os.makedirs('training_dataset', exist_ok=True)
num_videos = 1000
frames_per_video = 10
img_size = (64, 64)

prompts_and_movements = [
    ("circle moving down", "circle", "down"),
    ("circle moving left", "circle", "left"),
    # More movement directions...
]

def create_image_with_moving_shape(size, frame_num, shape, direction):
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Code to move the shape in different directions
    # ...

    return np.array(img)

for video_num in range(num_videos):
    prompt, shape, direction = random.choice(prompts_and_movements)
    video_frames = []
    for frame_num in range(frames_per_video):
        img_array = create_image_with_moving_shape(img_size, frame_num, shape, direction)
        video_frames.append(img_array)

    # Save each frame as an image
    video_dir = os.path.join('training_dataset', f'video_{video_num}')
    os.makedirs(video_dir, exist_ok=True)
    for frame_num, frame in enumerate(video_frames):
        frame_image = Image.fromarray(frame)
        frame_image.save(os.path.join(video_dir, f'frame_{frame_num}.png'))

    # Save the text prompt
    with open(f'{video_dir}/prompt.txt', 'w') as f:
        f.write(prompt)
```

### Step 2: GAN Model Definitions

We define the GAN components: the Generator and Discriminator. The Generator takes a noise vector and a text embedding and generates video frames, while the Discriminator classifies real versus generated frames.

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, text_embed_size):
        super(Generator, self).__init__()
        # Layers for upsampling noise and text embeddings
        # ...

    def forward(self, noise, text_embed):
        # Generator forward pass
        # ...

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Layers for downsampling images and classifying
        # ...

    def forward(self, input):
        # Discriminator forward pass
        # ...
```

### Step 3: Training the GAN

The training loop alternates between updating the Discriminator and Generator. The Generator creates fake video frames conditioned on text prompts, and the Discriminator distinguishes real frames from generated ones.

```python
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(embed_size=10).to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss().to(device)

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for real_data, prompts in dataloader:
        # Train Discriminator
        # Train Generator
        # Save models and generate video samples
```

### Step 4: Generating Videos from Text

After training, the model can generate videos based on a text prompt.

```python
def generate_video(text_prompt, num_frames=10):
    # Code to generate video frames based on text prompt
    # Save each frame as an image
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/texttovideo.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment and run the video dataset creation script.

## Contributing

If you have suggestions or find bugs, feel free to open issues or submit pull requests.

---

TextToVideo is an exciting project that bridges the gap between text and video generation, offering a creative tool for video synthesis.