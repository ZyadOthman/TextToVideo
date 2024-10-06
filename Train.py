import os
import torch
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib as plt
from IPython.display import clear_output, display, HTML
import base64
from TextToVideoDataset import TextToVideoDataset
from TextEmbedding import TextEmbedding
from Generator import Generator
from Discriminator import Discriminator
from DatasetGenerator import prompts_and_movements


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = TextToVideoDataset(root_dir='training_dataset', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

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
netD = Discriminator().to(device)
netD.load_state_dict(torch.load("discriminator.pth", map_location=device, weights_only=True))
criterion = nn.BCELoss().to(device)
optimizerD = optim.Adam(netD.parameters(),lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(),lr=0.0002, betas=(0.5, 0.999))

num_epochs = 400

for epoch in range(num_epochs):
    epoch_iterator = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
    
    for i, (data, prompts) in enumerate(epoch_iterator):
        real_data = data.to(device)
        prompts = [prompt for prompt in prompts]

        netD.zero_grad()
        batch_size = real_data.size(0)
        labels = torch.ones(batch_size, 1).to(device)
        output = netD(real_data)
        lossD_real = criterion(output, labels)
        lossD_real.backward()

        noise = torch.randn(batch_size, 100).to(device)
        text_embeds = torch.stack([text_embedding(encode_text(prompt).to(device)).mean(dim=0) for prompt in prompts])
        fake_data = netG(noise, text_embeds)
        labels = torch.zeros(batch_size, 1).to(device)
        output = netD(fake_data.detach())
        lossD_fake = criterion(output, labels)
        lossD_fake.backward()
        optimizerD.step()

        netG.zero_grad()
        labels = torch.ones(batch_size, 1).to(device)
        output = netD(fake_data)
        lossG = criterion(output, labels)
        lossG.backward()
        optimizerG.step()

        epoch_iterator.set_postfix(Loss_D=(lossD_real + lossD_fake).item(), Loss_G=lossG.item())

    if (epoch + 1) % 5 == 0:
        torch.save(netG.state_dict(), f'generator{epoch}.pth')
        torch.save(netD.state_dict(), f'discriminator{epoch}.pth')
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss D: {lossD_real + lossD_fake}, Loss G: {lossG}")


torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')

def generate_video(text_prompt, num_frames=10):
    os.makedirs(f'generated_video_{text_prompt.replace(" ", "_")}', exist_ok=True)
    
    text_embed = text_embedding(encode_text(text_prompt).to(device)).mean(dim=0).unsqueeze(0)
    
    for frame_num in range(num_frames):
        noise = torch.randn(1, 100).to(device)
        
        with torch.no_grad():
            frame = torch.tensor([frame_num], dtype=torch.float).to(device)
            fake_frame = netG(noise, text_embed)
        
        save_image(fake_frame, f'generated_video_{text_prompt.replace(" ", "_")}/frame_{frame_num}.png')

generate_video('circle moving up-right')

folder_path = 'generated_video_circle_moving_up-right'


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

