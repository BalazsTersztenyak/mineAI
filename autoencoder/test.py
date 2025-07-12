from minedojo.data import YouTubeDataset
import yt_dlp
import cv2
import pandas as pd

import os
from PIL import Image

df = pd.read_json('data/dataset/youtube_full.json')

ydl_opts = {
        'format': 'bestvideo[height=360][ext=mp4]',  # Selects only 360p mp4 video
        'outtmpl': f'data/dataset/videos/%(title)s.%(ext)s',
        'merge_output_format': 'mp4'
    }
url = df['link'][201_100]
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    success = True
except:
    print(f"Video {url} unavailable, skipping")

# try:
#     yt = YouTube(df['link'][200_000])
#     video = yt.streams.filter(res="360p").first()
# except:
#     print(f"Video {df['link'][200_000]} unavailable, skipping")
# else:
#     print(f"Video {df['link'][200_000]} downloading")
#     video.download('data/dataset/videos')

# video = 'A TOMAR POR SACO EL MAPA! Minecraft PARKOUR PARADISE 2 con MAZAFESIA! Cap11! FINAL!.mp4'
video = os.listdir('data/dataset/videos')[0]
# clip = VideoFileClip(os.path.join('data/dataset/videos', video))
cap = cv2.VideoCapture(os.path.join('data/dataset/videos', video))

# Initialize frame count
count = 0

# Iterate over each frame in the video clip
while cap.isOpened():
    # Save only every Nth frame
    ret, frame = cap.read()
    if count % 1000 != 0:
    # if count < 19000 :
        count += 1
        continue
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    path = f"data/dataset/test_frames/test_{count}.png"
    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(frame)

    # Save the PIL Image as PNG
    pil_image.save(path)
    count += 1
    # break
