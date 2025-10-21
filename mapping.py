# Main mapping file that calls the other functions
# Runs the camera and takes the image, then calls image_stitcher
import cv2
import numpy as np
import ImageStitcher
import time
import os

directory = "images/"

# TODO: Use opencv for realtime image stitching
if __name__ == "__main__":
    stitcher = ImageStitcher()

    # '0' designates default camera, may need to change
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        stitcher.add_image(frame)

        # Placeholder, change for multithreading
        time.sleep(3)

    map = stitcher.get_map()
    cv2.imwrite(os.path.join(directory, 'aerial_map.png'), map)
