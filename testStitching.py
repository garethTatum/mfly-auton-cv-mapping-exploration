# Main mapping file that calls the other functions
# Runs the camera and takes the image, then calls image_stitcher
import cv2
import numpy as np
from ImageStitcher import ImageStitcher
import time
import os

directory = "/home/gtatum/mfly_cv/mfly-auton-cv-mapping-exploration/images/Portion"

# TODO: Use opencv for realtime image stitching
if __name__ == "__main__":
    stitcher = ImageStitcher()

    imgs = []

    for image in os.listdir(directory):
        # check if the image ends with jpg
        if (image.endswith(".JPG")):
            img = cv2.imread(directory + "/" + image)
            imgs.append(img)

    stitcher.stitch_images(imgs)
    map = stitcher.get_map()
    cv2.imwrite(os.path.join(directory, 'aerial_map.png'), map)
