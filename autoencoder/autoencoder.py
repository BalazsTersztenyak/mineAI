import torch
# from torch.nn import HuberLoss
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
import pandas as pd
import yt_dlp
import os
import cv2
from minedojo.data import YouTubeDataset
# from ..models.encoder import Autoencoder
import shutil
from tqdm import tqdm
import math
# from models.encoder_dataset import EncoderDataset, TRANSFORM
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
    # model, criterion, optimizer = model_setup(device, model_path)

    idx = 1
    while video < dataset.shape[0]:
        print(f"Training video {video}")
        idx = prepare_video(dataset, idx) + 1
        break
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

def prepare_video(df, idx = 0):
    """
    Function to download videos and process frames. It takes the number of videos to download as input and does not return anything.
    """

    # Prepare folders
    if os.path.exists(FRAMES_PATH):
        frames = os.listdir(FRAMES_PATH)
        for frame in frames:
            os.remove(os.path.join(FRAMES_PATH, frame))

    if os.path.exists(VIDEO_PATH):
        videos = os.listdir(VIDEO_PATH)
        for video in videos:
            os.remove(os.path.join(VIDEO_PATH, video))

    os.makedirs(FRAMES_PATH, exist_ok=True)
    os.makedirs(VIDEO_PATH, exist_ok=True)

    idx, url, frac_part = download_video(df, idx)
    prepare_images(url, frac_part)
    return idx

def download_video(df, idx):
    success = False
    ydl_opts = {
        'format': 'bestvideo[height=360][ext=mp4]',  # Selects only 360p video
        'outtmpl': f'{VIDEO_PATH}/%(title)s.%(ext)s',
        'merge_output_format': 'mp4'
    }
    while not success:
        # Get video URLs based on progress
        url = df['link'][idx]
        url = 'https://www.youtube.com/watch?v=GnJimSWdNXI'
        length = df['duration'][idx]
        fps = df['fps'][idx]
        n_frames = math.floor(length * fps)
        if n_frames < N_FRAME_PER_VID:
            frac_part = 1
        else:
            frac_part = math.floor(n_frames / N_FRAME_PER_VID)
        # print(length, fps, n_frames, frac_part)
        # Download videos and process frames
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            success = True
            # yt = YouTube(url)
            # video = yt.streams.filter(res="360p").first()
        except:
            print(f"Video {url} unavailable, skipping")
            idx += 1

    return idx, url, frac_part
    
def prepare_images(url, frac_part):
    # Iterate through downloaded videos
    video = os.listdir(VIDEO_PATH)
    assert len(video) == 1
    video = video[0]
    # print(video)
    cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video))
    # ret, frame = cap.read()
    # print(ret, frame)
    
    frame_number = 0
    with tqdm(total=N_FRAME_PER_VID, desc=f"Creating images from video: {url}") as pbar:
        while cap.isOpened():
            # print("stepped inside")
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % frac_part == 0:
                cv2.imwrite(f"{FRAMES_PATH}/frame_{frame_number:04d}.jpg", frame)
                pbar.update(1)
            frame_number += 1
            if frame_number * frac_part >= N_FRAME_PER_VID:
                break
    cap.release()
    print("Frames extracted successfully!")
    
    # Remove downloaded video
    os.remove(os.path.join(VIDEO_PATH, video))

    return

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