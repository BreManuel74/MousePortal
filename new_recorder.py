import os
import cv2
import numpy as np
import pymmcore_plus
import time

def main():
    # Initialize the Micro-Manager core
    mmc = pymmcore_plus.CMMCorePlus()
    mmc.loadSystemConfiguration(r'C:\Program Files\Micro-Manager-2.0\ThorCam.cfg')
    camera_device = 'ThorCam'
    mmc.setCameraDevice(camera_device)
    
    # Debug: List available properties
    #print("Available properties for ThorCam:")
    #print(mmc.getDevicePropertyNames(camera_device))
    
    # # Enable hardware sequencing for hardware triggering
    # mmc.mda.engine.use_hardware_sequencing = True
    # print("Hardware sequencing enabled for hardware triggering.")
    
    # Video output settings
    out_filename = f"video_{int(time.time())}.avi"
    frame_width = int(mmc.getImageWidth())
    frame_height = int(mmc.getImageHeight())
    fps = 30  # Frames per second for the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height), isColor=False)
    
    # Start hardware-triggered sequence acquisition
    num_frames = 0  # 0 means indefinite acquisition until stopped manually

    stop_file = "stop_recording.flag"

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
        print(f"Recording stopped on {camera_device}. Video saved.")

if __name__ == "__main__":
    main()