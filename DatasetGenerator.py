import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.makedirs('training_dataset', exist_ok=True)
num_videos = 1000
frames_per_video = 10
img_size = (64, 64)
shape_size = 10

prompts_and_movements = [
    ("circle moving down", "circle", "down"),
    ("circle moving left", "circle", "left"),
    ("circle moving right", "circle", "right"),
    ("circle moving diagonally up-right", "circle", "diagonal_up_right"),
    ("circle moving diagonally down-left", "circle", "diagonal_down_left"),
    ("circle moving diagonally up-left", "circle", "diagonal_up_left"),
    ("circle moving diagonally down-right", "circle", "diagonal_down_right"),
    ("circle bouncing vertically", "circle", "bounce_vertical"),
    ("circle bouncing horizontally", "circle", "bounce_horizontal"),
    ("circle zigzagging vertically", "circle", "zigzag_vertical"),
    ("circle zigzagging horizontally", "circle", "zigzag_horizontal"),
    ("circle moving up-left", "circle", "up_left"),
    ("circle moving down-right", "circle", "down_right"),
    ("circle moving down-left", "circle", "down_left")
]

def create_image_with_moving_shape(size, frame_num, shape, direction):
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2
    
    if direction == "down":
        position = (center_x, (center_y + frame_num * 5) % size[1])
    elif direction == "left":
        position = ((center_x - frame_num * 5) % size[0], center_y)
    elif direction == "right":
        position = ((center_x + frame_num * 5) % size[0], center_y)
    elif direction == "diagonal_up_right":
        position = ((center_x + frame_num * 5) % size[0], (center_y - frame_num * 5) % size[1])
    elif direction == "diagonal_down_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    elif direction == "diagonal_up_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y - frame_num * 5) % size[1])
    elif direction == "diagonal_down_right":
        position = ((center_x + frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    elif direction == "bounce_vertical":
        position = (center_x, center_y - abs(frame_num * 5 % size[1] - center_y))
    elif direction == "bounce_horizontal":
        position = (center_x - abs(frame_num * 5 % size[0] - center_x), center_y)
    elif direction == "zigzag_vertical":
        position = (center_x, center_y - frame_num * 5 % size[1] if frame_num % 2 == 0 else center_y + frame_num * 5 % size[1])
    elif direction == "zigzag_horizontal":
        position = (center_x - frame_num * 5 % size[0] if frame_num % 2 == 0 else center_x + frame_num * 5 % size[0], center_y)
    elif direction == "up_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y - frame_num * 5) % size[1])
    elif direction == "down_right":
        position = ((center_x + frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    elif direction == "down_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    else:
        position = (center_x, center_y)

    shape_size = 10
    if shape == "circle":
        draw.ellipse([position[0] - shape_size // 2, position[1] - shape_size // 2, position[0] + shape_size // 2, position[1] + shape_size // 2], fill=(0, 0, 0))

    return np.array(img)


for video_num in range(num_videos):
    prompt, shape, direction = random.choice(prompts_and_movements)
    video_frames = []
    for frame_num in range(frames_per_video):
        img_array = create_image_with_moving_shape(img_size, frame_num, shape, direction)
        video_frames.append(img_array)

    video_dir = os.path.join('training_dataset', f'video_{video_num}')
    os.makedirs(video_dir, exist_ok=True)
    for frame_num, frame in enumerate(video_frames):
        frame_image = Image.fromarray(frame)
        frame_image.save(os.path.join(video_dir, f'frame_{frame_num}.png'))

for i in range(num_videos):
    prompt, shape, direction = random.choice(prompts_and_movements)
    video_dir = f'training_dataset/video_{i}'
    os.makedirs(video_dir, exist_ok=True)

    with open(f'{video_dir}/prompt.txt', 'w') as f:
        f.write(prompt)

    for frame_num in range(frames_per_video):
        img = create_image_with_moving_shape(img_size, frame_num, shape, direction)
        cv2.imwrite(f'{video_dir}/frame_{frame_num}.png', img)