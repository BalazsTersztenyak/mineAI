import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from pytube import YouTube
from PIL import Image
import os
from moviepy.editor import VideoFileClip
import time
import requests
from minedojo.data import YouTubeDataset
import math

# Define the dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image
        except:
            print(f"Couldn't load image: {image_path}")

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        """
        Initialize the Autoencoder class with encoder and decoder layers using nn.Sequential and nn.Conv2d/nn.ConvTranspose2d.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # b, 16, 180, 320
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b, 32, 90, 160
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 45, 80
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # b, 32, 90, 160
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # b, 16, 180, 320
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # b, 3, 360, 640
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Perform the forward pass through the encoder and decoder and return the result.
        
        Args:
            x: Input data to be processed.
            
        Returns:
            The output of the forward pass.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def download_videos():
    """
    Function to download videos and process frames. It takes the number of videos to download as input and does not return anything.
    """

    response = requests.get("https://zenodo.org/records/6693792/files/youtube_full.json?download=1")

    with open('data/dataset/youtube_full.json', 'wb') as f:
        f.write(response.content)

    # Check progress file and set videos_done
    try:
        with open('progress', 'r') as file:
            videos_done = int(file.readline())
    except:
        with open('progress', 'w') as file:
            file.write('0')
            videos_done = 0

    # Get video URLs based on progress
    urls = df['link'][videos_done:videos_done+n_vids]

    # Download videos and process frames
    for url in urls:
        try:
            yt = YouTube(url)
        except:
            print(f"Video {url} unavailable, skipping")
        else:
            print(f"Video {url} downloading")
            video = yt.streams.filter(res="360p").first()
            video.download('./data/dataset/videos')
    
    start = time.time()

    # Iterate through downloaded videos
    for video in os.listdir('./data/dataset/videos'):
        clip = VideoFileClip(os.path.join('./data/dataset/videos', video))

        # Initialize frame count
        count = 0

        print(f"Converting video: {video}")

        # Iterate over each frame in the video clip
        for frame in clip.iter_frames():
            # Save only every Nth frame
            if count % n_part != 0:
                count += 1
                continue

            path = f"./data/dataset/frames/{video}_{count}.png"
            
            # Skip if it exists (used only during testing)
            if os.path.exists(path):
                count += 1
                continue

            # Convert the frame to a PIL Image
            pil_image = Image.fromarray(frame)

            # Save the PIL Image as PNG
            pil_image.save(path)

            # Log every n_part*100th
            if count % (100 * n_part) == 0:
                now = time.time()
                diff = math.round(now - start, 2)
                start = now
                print(f"Video: {video}, frame: {count}, time: {diff}")

            count += 1

        # Close the video clip
        clip.close()
        print(f'Video {video} sliced. Saved frames: {math.floor(count / n_part)}')

        # Remove downloaded video
        os.remove(os.path.join('./data/dataset/videos', video))

    # Update progress file
    with open('progress', 'w') as file:
        file.write(str(videos_done+n_vids))

def model_setup():
    """
    Function to initialize the autoencoder, define the loss function and optimizer, and return the model, criterion, and optimizer.
    """

    print('Start model setup')

    global model, criterion, optimizer, train_loader

    model = Autoencoder()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    # Assuming you have your dataset loaded into 'images' variable as a list of PIL images
    train_dataset = CustomDataset('./data/dataset/frames', transform=transform)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print('Model setup Done')

def train(cycle=0):
    """
    Trains a model using the given train_loader, criterion, and optimizer for the specified number of epochs. Optionally, it can also specify the cycle number. 

    Args:
        model: The model to be trained.
        train_loader: The data loader for training data.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        num_epochs: The number of epochs to train the model.
        cycle (optional): The cycle number (default is 0).

    Returns:
        None
    """

    print('Train started...')

    start = time.time()

    # Training
    total_step = len(train_loader)
    losses = []
    for epoch in range(num_epochs):
        for i, images in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            losses.append(loss)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                now = time.time()
                diff = math.round(now - start, 2)
                start = now

                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), diff))
        
                   
    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join('./data/checkpoints', f'model_{cycle}.ckpt'))
    print('Model saved')

    # Save the losses
    with open(os.path.join('./data/checkpoints', f'losses_{cycle}.txt'), 'w') as file:
        for loss in losses:
            file.write(str(loss.item()) + '\n')

    # Remove the frames folder
    os.rmdir(train_loader.dataset.folder_path)
    
def setup():
    global df, num_epochs, batch_size, learning_rate, n_part, n_vids, transform
    
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

    # Hyperparameters
    num_epochs = 1
    batch_size = 32
    learning_rate = 0.001
    n_part = 5
    n_vids = 1

    # Making the code device-agnostic
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # Define data augmentation transforms
    transform = transforms.Compose([
        transforms.Resize((360, 640)),  # Resize images to 640x360
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),
    ])

def train_loop():
    cycle = 0

    while cycle < 1: #df.shape[0]:
        print(f"Training cycle {cycle}")
        download_videos()
        model_setup()
        train(cycle)
        cycle += 1

def main():
    setup()
    train_loop()

if __name__ == '__main__':
    main()