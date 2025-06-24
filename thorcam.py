import os
import cv2
import numpy as np
import pymmcore_plus
import time
from Phases.user import Stopwatch

def main():
    global_stopwatch = Stopwatch()
    global_stopwatch.start()

    camera_device = "ThorCam"
    video_dir = os.environ.get("OUTPUT_DIR")
    fps = 30
    stop_file = "stop_recording.flag"

    # Initialize the Micro-Manager core
    mmc = pymmcore_plus.CMMCorePlus()
    mmc.loadSystemConfiguration(r'C:\Program Files\Micro-Manager-2.0\ThorCam.cfg')
    mmc.setCameraDevice(camera_device)
    # Set camera exposure lower to make the image dimmer
    mmc.setProperty(camera_device, "Exposure", 6)  # Set to your desired value in ms (e.g., 1 for minimum)

    #print(mmc.getDevicePropertyNames(camera_device))

    # Video output settings
    os.makedirs(video_dir, exist_ok=True)
    out_filename = os.path.join(video_dir, f"{int(time.time())}pupil_cam.avi")
    frame_width = int(mmc.getImageWidth())
    frame_height = int(mmc.getImageHeight())
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Prepare text log file
    log_path = os.path.join(video_dir, f"{int(time.time())}_frame_log.txt")
    log_file = open(log_path, "w")
    log_file.write("time_seconds\tframe_number\n")

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

                # --- Live view window (no quit feature) ---
                cv2.imshow("Live View", frame.astype(np.uint8))
                cv2.waitKey(1)
                # ------------------------------------------

                video_writer.write(frame.astype(np.uint8))  # Write frame to video
                num_frames += 1
                # Write time and frame number to text file
                log_file.write(f"{global_stopwatch.get_elapsed_time():.2f}\t{num_frames}\n")
                log_file.flush()
            else:
                time.sleep(0.01)  # Prevent busy-waiting
    finally:
        mmc.stopSequenceAcquisition()
        video_writer.release()  # Release the video writer
        log_file.close()
        cv2.destroyAllWindows()  # Close the live view window
        print(f"Recording stopped on {camera_device}. Video saved at {out_filename}.")
        print(f"Frame log saved at {log_path}")

if __name__ == "__main__":
    main()