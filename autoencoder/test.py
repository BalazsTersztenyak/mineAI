from minedojo.data import YouTubeDataset
from pytube import YouTube
import pandas as pd

import os
from moviepy.editor import VideoFileClip
from PIL import Image

df = pd.read_json('data/dataset/youtube_full.json')

try:
    yt = YouTube(df['link'][200_000])
    video = yt.streams.filter(res="360p").first()
except:
    print(f"Video {df['link'][200_000]} unavailable, skipping")
else:
    print(f"Video {df['link'][200_000]} downloading")
    video.download('data/dataset/videos')

# video = 'A TOMAR POR SACO EL MAPA! Minecraft PARKOUR PARADISE 2 con MAZAFESIA! Cap11! FINAL!.mp4'
video = os.listdir('data/dataset/videos')[0]
clip = VideoFileClip(os.path.join('data/dataset/videos', video))

# Initialize frame count
count = 0

# Iterate over each frame in the video clip
for frame in clip.iter_frames():
    # Save only every Nth frame
    if count % 500 != 0:
        count += 1
        continue

    path = f"data/dataset/test_frames/test_{count}.png"
    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(frame)

    # Save the PIL Image as PNG
    pil_image.save(path)
    count += 1