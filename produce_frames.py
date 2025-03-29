import cv2
import os

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

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        filename = os.path.join(output_folder, f"frame_{second}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Guardado: {filename}")

        second += 1
        frame_idx += interval  

    cap.release()
    print("Proceso terminado.")

video_path = "video.mp4"  
output_folder = "frames"  
produce_frames_1hz(output_folder)
