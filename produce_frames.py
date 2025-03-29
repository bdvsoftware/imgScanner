import cv2
import os
import time

def produce_frames_1hz(output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture("video/videoplayback.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"FPS: {fps}, Duración: {duration:.2f} segundos")


    interval = int(fps)

    frame_idx = 0
    second = 0
    index = 10000

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        filename = os.path.join(output_folder, f"{index}_frame_{second}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        second += 1
        frame_idx += interval  
        index += 1

    cap.release()
    print("END")

video_path = "video.mp4"  
output_folder = "frames"  
produce_frames_1hz(output_folder)
