# Main mapping file that calls the other functions
# Runs the camera and takes the image, then calls image_stitcher
import cv2
import numpy as np
from ImageStitcher import ImageStitcher
import time
import os

directory = "images/small"

# TODO: Use opencv for realtime image stitching
if __name__ == "__main__":
    stitcher = ImageStitcher()

    sorted_files = sorted(os.listdir(directory))

    imgs = []

    for image in sorted_files:
        # check if the image ends with jpg
        if (image.lower().endswith(".jpg")):
            img = cv2.imread(directory + "/" + image)
            imgs.append(img)

    stitcher.stitch_images(imgs)
    map = stitcher.get_map()
    cv2.imwrite(os.path.join(directory, 'aerial_map.png'), map)
