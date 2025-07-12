import yt_dlp

def download_video_360p(video_url, output_path='.'):
    ydl_opts = {
        'format': 'bestvideo[height=360][ext=mp4]',  # Selects only 360p video
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Example usage
video_url = 'https://www.youtube.com/watch?v=GnJimSWdNXI'
download_video_360p(video_url, output_path='data/dataset/videos')

import cv2

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number == 100:
            cv2.imwrite(f"{output_folder}/frame_{frame_number:04d}.jpg", frame)
            
        frame_number += 1
    
    cap.release()
    print("Frames extracted successfully!")

# Example usage
extract_frames("data/dataset/videos/Nature's Beauty - Ep. 11： Canvas.mp4", "data/dataset/frames")