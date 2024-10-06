import os
import cv2
import torch
import numpy as np
from torchvision.utils import save_image
from DatasetGenerator import prompts_and_movements
from Generator import Generator
from TextEmbedding import TextEmbedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_prompts = [prompt for prompt, _, _ in prompts_and_movements]
vocab = {word: idx for idx, word in enumerate(set(" ".join(all_prompts).split()))}
vocab_size = len(vocab)
embed_size = 10

def encode_text(prompt):
    return torch.tensor([vocab[word] for word in prompt.split()])

text_embedding = TextEmbedding(vocab_size, embed_size).to(device)

netG = Generator(embed_size).to(device)
netG.load_state_dict(torch.load("generator.pth", map_location=device, weights_only=True))

def generate_video(text_prompt, num_frames=10):
    os.makedirs(f'generated_video_{text_prompt.replace(" ", "_")}', exist_ok=True)
    
    text_embed = text_embedding(encode_text(text_prompt).to(device)).mean(dim=0).unsqueeze(0)
    
    for frame_num in range(num_frames):
        noise = torch.randn(1, 100).to(device)
        
        with torch.no_grad():
            fake_frame = netG(noise, text_embed)
        
        save_image(fake_frame, f'generated_video_{text_prompt.replace(" ", "_")}/frame_{frame_num}.png')

generate_video('circle bouncing vertically')

folder_path = 'generated_video_circle_bouncing_vertically'


image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

image_files.sort()

frames = []

for image_file in image_files:
  image_path = os.path.join(folder_path, image_file)
  frame = cv2.imread(image_path)
  frames.append(frame)

frames = np.array(frames)

fps = 10

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('generated_video.avi', fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

for frame in frames:
  out.write(frame)

out.release()