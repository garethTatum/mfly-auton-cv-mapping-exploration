# Main mapping file that calls the other functions
# Runs the camera and takes the image, then calls image_stitcher
import cv2
import numpy as np
from multithreading import ImageStitcher
import time
import os

directory = "images/portion"

if __name__ == "__main__":
    stitcher = ImageStitcher()

    sorted_files = sorted(os.listdir(directory))

    imgs = []

    for image in sorted_files:
        # check if the image ends with jpg
        if (image.lower().endswith(".jpg") or image.lower().endswith(".jpeg")):
            img = cv2.imread(directory + "/" + image)
            imgs.append(img)

    stitcher.run_test_small(imgs, downsample=0.4)
    map = stitcher.get_mosaics()

    for i in range(len(map)):
        cv2.imwrite(os.path.join(directory, 'aerial_map_multi_roi_blended_' + str(i) + '.png'), map[i])
