import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 0
    
    while success:
        if count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame{count}.jpg"), image)
        success, image = video.read()
        count += 1

    video.release()

# Input paths
input_path = "C:/Users/sukes/Downloads/traffic_images.mp4"  # Path to a video file or directory
frame_directory = "C:/Users/sukes/OneDrive/Desktop/360/project_2"

# Check if the input path is a file or a directory
if os.path.isfile(input_path):  # Single video file
    video_path = input_path
    output_dir = os.path.join(frame_directory, "frames_" + os.path.basename(video_path)[:-4])
    extract_frames(video_path, output_dir)

elif os.path.isdir(input_path):  # Directory containing multiple videos
    videos = [video for video in os.listdir(input_path) if video.endswith('.mp4')]
    for video in videos:
        video_path = os.path.join(input_path, video)
        output_dir = os.path.join(frame_directory, "frames_" + video[:-4])
        extract_frames(video_path, output_dir)

else:
    print(f"Invalid path: {input_path}")
