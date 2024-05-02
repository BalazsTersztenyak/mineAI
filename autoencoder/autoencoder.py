import torch
from torch.nn import MSELoss, L1Loss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
# import numpy as np
from pytube import YouTube
from PIL import Image
import os
from moviepy.editor import VideoFileClip
from minedojo.data import YouTubeDataset
from models.encoder import Autoencoder
import shutil
from tqdm import tqdm
import math
from accelerate import Accelerator

# Hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_FRAME_PER_VID = 4096

# Define data augmentation transforms
TRANSFORM = transforms.Compose([
    transforms.Resize((360, 640)),  # Resize images to 640x360
    transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),  # Randomly translate the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
    transforms.ToTensor()
])

# Define the dataset
class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = TRANSFORM

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image).to('cuda')

            return image
        except:
            print(f"Couldn't load image: {image_path}")

# Download dataset
def download_videos(idx = 0):
    """
    Function to download videos and process frames. It takes the number of videos to download as input and does not return anything.
    """

    os.makedirs('./data/dataset/frames', exist_ok=True)
    os.makedirs('./data/dataset/videos', exist_ok=True)
    success = False
    while not success:

        # Get video URLs based on progress
        url = df['link'][idx]
        length = df['duration'][idx]
        fps = df['fps'][idx]
        n_frames = math.floor(length * fps)
        frac_part = math.floor(n_frames / N_FRAME_PER_VID)
        
        # Download videos and process frames
        try:
            yt = YouTube(url)
            print(f"Video {url} downloading")
            video = yt.streams.filter(res="360p").first()
        except:
            print(f"Video {url} unavailable, skipping")
            idx += 1
        else:
            video.download('./data/dataset/videos')
            success = True
    
    # Iterate through downloaded videos
    for video in os.listdir('./data/dataset/videos'):
        clip = VideoFileClip(os.path.join('./data/dataset/videos', video))

        print(f"Converting video: {video}")
        counter = 0
        n_saved = 0
        with tqdm(total=N_FRAME_PER_VID, desc=f"Video: {url}") as pbar:
            # Iterate over each frame in the video clip
            for frame in clip.iter_frames():
                # Save only every Nth frame
                if counter % frac_part != 0:
                    counter += 1
                    continue
                
                path = f"./data/dataset/frames/{video}_{counter}.png"

                # Skip if it exists (used only during testing)
                if os.path.exists(path):
                    counter += 1
                    n_saved += 1
                    pbar.update(1)
                    continue

                # Convert the frame to a PIL Image
                pil_image = Image.fromarray(frame)

                # Save the PIL Image as PNG
                pil_image.save(path)

                counter += 1
                n_saved += 1

                if n_saved >= N_FRAME_PER_VID:
                    break

                pbar.update(1)

        # Close the video clip
        clip.close()
        print(f'Video {video} sliced.')

        # Remove downloaded video
        os.remove(os.path.join('./data/dataset/videos', video))

        # Update progress file
        with open('progress', 'w') as file:
            file.write(str(idx+1))

    return idx

def model_setup(mse):
    """
    Function to initialize the autoencoder, define the loss function and optimizer, and return the model, criterion, and optimizer.
    """

    print('Start model setup')

    global model

    model = Autoencoder().to(device)
    model.summary()
    model = accelerator.prepare(model)

def update_model(mse):
    global criterion, optimizer, train_loader

    # Loss function and optimizer
    if(mse):
        criterion = MSELoss()
    else:
        criterion = L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = accelerator.prepare(optimizer)

    # Assuming you have your dataset loaded into 'images' variable as a list of PIL images
    train_dataset = CustomDataset('./data/dataset/frames')

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('Model setup Done')

def train(video):
    """
    Trains a model using the given train_loader, criterion, and optimizer for the specified number of epochs. Optionally, it can also specify the video number. 

    Args:
        model: The model to be trained.
        train_loader: The data loader for training data.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        NUM_EPOCHS: The number of epochs to train the model.
        video (optional): The video number.

    Returns:
        None
    """

    print('Train started...')

    os.makedirs('./data/checkpoints', exist_ok=True)

    # Training
    total_step = len(train_loader)
    losses = []
    for epoch in range(NUM_EPOCHS):
        with tqdm(total=len(train_loader), desc=f"Training on video {video}: ") as pbar:
            for images in train_loader:
                # Forward pass
                images = accelerator.prepare(images.to('cuda')) 
                outputs = model(images)
                outputs = outputs.to('cpu')
                images = images.to('cpu')
                loss = criterion(outputs, images)
                losses.append(loss)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}')
                   
    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join('./data/checkpoints', f'model_{video}.ckpt'))
    print('Model saved')

    # Save the losses
    with open(os.path.join('./data/checkpoints', f'losses_{video}.txt'), 'w') as file:
        for loss in losses:
            file.write(str(loss.item()) + '\n')

    # Remove the frames folder
    shutil.rmtree(train_loader.dataset.folder_path)
    
def setup():
    global df, device, accelerator
    
    # Download dataset
    youtube_dataset = YouTubeDataset(
      full=True,          # full=False for tutorial videos or 
                 # full=True for general gameplay videos
      download=True,        # download=True to automatically download data or 
                 # download=False to load data from download_dir
      download_dir='data/dataset'
                 # default: "~/.minedojo". You can also manually download data from
                 # https://doi.org/10.5281/zenodo.6641142 and put it in download_dir.           
    )

    df = pd.read_json('./data/dataset/youtube_full.json')

    # Making the code device-agnostic
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using {device}")

def train_loop():
    video = 0
    switch_rate = 50
    mse = True
    model_setup(mse)
    idx = 0
    while video < 4*switch_rate: #df.shape[0]:
        print(f"Training video {video}")
        idx = download_videos(idx) + 1
        update_model(mse)
        train(video)
        video += 1
        if(video % switch_rate == 0):
            mse = not mse

def main():
    setup()
    train_loop()

if __name__ == '__main__':
    main()