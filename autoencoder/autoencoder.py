import torch
from torch.nn import HuberLoss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pytube import YouTube
from PIL import Image
import os
from moviepy import VideoFileClip
from minedojo.data import YouTubeDataset
from ..models.encoder import Autoencoder
import shutil
from tqdm import tqdm
import math
from models.encoder_dataset import EncoderDataset, TRANSFORM
import pickle
# TODO : rework the imports
# Hyperparameters
N_EPOCH_PER_VIDEO = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_FRAME_PER_VID = 4096
VIDEO_PATH = './data/dataset/videos'
FRAMES_PATH = './data/dataset/frames'
CHECKPOINTS_PATH = './data/checkpoints'
 
def main():
    dataset, device = setup()
    train_loop(dataset, device)

def setup():
    if not os.path.exists('./data/dataset/youtube_full.json'):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    return df, device

def train_loop(dataset, device):
    model_path = None
    video = 0
    model, criterion, optimizer = model_setup(device, model_path)

    idx = 0
    while video < dataset.shape[0]:
        print(f"Training video {video}")
        idx = download_videos(dataset, idx) + 1
        train_loader = update_dataloader()
        model, losses = train(video, train_loader, model, criterion, optimizer, device)
        save_model(video, model, losses)
        video += 1

def model_setup(device, model_path = None):
    print('Start model setup')

    model = Autoencoder().to(device)
    model.summary()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    criterion = HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer

def download_videos(df, idx = 0):
    """
    Function to download videos and process frames. It takes the number of videos to download as input and does not return anything.
    """

    os.makedirs(FRAMES_PATH, exist_ok=True)
    os.makedirs(VIDEO_PATH, exist_ok=True)

    success = False
    while not success:
        # Get video URLs based on progress
        url = df['link'][idx]
        length = df['duration'][idx]
        fps = df['fps'][idx]
        n_frames = math.floor(length * fps)
        if n_frames < N_FRAME_PER_VID:
            frac_part = 1
        else:
            frac_part = math.floor(n_frames / N_FRAME_PER_VID)
        
        # Download videos and process frames
        try:
            yt = YouTube(url)
            video = yt.streams.filter(res="360p").first()
        except:
            print(f"Video {url} unavailable, skipping")
            idx += 1
        else:
            print(f"Video {url} downloading")
            video.download(VIDEO_PATH)
            success = True
    
    # Iterate through downloaded videos
    for video in os.listdir(VIDEO_PATH):
        clip = VideoFileClip(os.path.join(VIDEO_PATH, video))

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
                
                path = os.path.join(FRAMES_PATH, f"{video}_{counter}.png")

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
        os.remove(os.path.join(VIDEO_PATH, video))

    return idx

def update_dataloader():
    # Assuming you have your dataset loaded into 'images' variable as a list of PIL images
    train_dataset = EncoderDataset(FRAMES_PATH) # TODO : rewrite the EncoderDataset class 

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) # TODO : rewrite the DataLoader class

    print('DataLoader setup Done')

    return train_loader

def train(video, train_loader, model, criterion, optimizer, device):
    print('Train started...')

    # Training
    total_step = len(train_loader)
    losses = []
    for epoch in range(N_EPOCH_PER_VIDEO):
        with tqdm(total=len(train_loader), desc=f"Training on video {video}: ") as pbar:
            for images in train_loader:
                # Forward pass
                images = images.to(device) 
                outputs = model(images)
                outputs = outputs.to('cpu')
                images = images.to('cpu')
                loss = criterion(outputs, images)
                losses.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        print(f'Epoch [{epoch+1}/{N_EPOCH_PER_VIDEO}], Loss: {loss.item()}')

    return model, losses

def save_model(video, model, losses):
    print('Saving model...')
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_PATH, f'model_{video}.ckpt'))
    print('Model saved')

    # Save the losses
    with open('loss_list', 'ab') as fp:
        pickle.dump(losses, fp)

    print('Losses saved')

    # Remove the frames folder
    shutil.rmtree(FRAMES_PATH)

if __name__ == '__main__':
    main()