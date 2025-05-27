import os
import cv2
import numpy as np
import pymmcore_plus
import time
import json

def load_config(config_path="cfg.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    cfg = load_config()

    # Get paths from config, with fallback defaults
    mm_config_path = cfg.get("mm_config_path", r'C:\Program Files\Micro-Manager-2.0\ThorCam.cfg')
    camera_device = cfg.get("camera_device", "ThorCam")
    video_dir = cfg.get("video_output_dir", ".")
    fps = cfg.get("video_fps", 30)
    fourcc_str = cfg.get("video_fourcc", "XVID")
    stop_file = cfg.get("stop_file", "stop_recording.flag")

    # Initialize the Micro-Manager core
    mmc = pymmcore_plus.CMMCorePlus()
    mmc.loadSystemConfiguration(mm_config_path)
    mmc.setCameraDevice(camera_device)

    # Video output settings
    os.makedirs(video_dir, exist_ok=True)
    out_filename = os.path.join(video_dir, f"{int(time.time())}pupil_cam.avi")
    frame_width = int(mmc.getImageWidth())
    frame_height = int(mmc.getImageHeight())
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Start hardware-triggered sequence acquisition
    num_frames = 0  # 0 means indefinite acquisition until stopped manually

    try:
        mmc.startSequenceAcquisition(num_frames, 0, True)
        print(f"Recording started on {camera_device}.")

        # Retrieve and save frames to the video
        while True:
            if os.path.exists(stop_file):
                print("Stop file detected. Terminating recording.")
                break
            if mmc.getRemainingImageCount() > 0:
                image = mmc.popNextImage()  # Retrieve the next image
                frame = np.reshape(image, (frame_height, frame_width))  # Reshape to 2D array
                video_writer.write(frame.astype(np.uint8))  # Write frame to video
                #print("Frame written to video.")
            else:
                time.sleep(0.01)  # Prevent busy-waiting
    finally:
        mmc.stopSequenceAcquisition()
        video_writer.release()  # Release the video writer
        print(f"Recording stopped on {camera_device}. Video saved at {out_filename}.")

if __name__ == "__main__":
    main()