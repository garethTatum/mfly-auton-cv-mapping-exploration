# Main mapping file that calls the other functions
# Runs the camera and takes the image, then calls image_stitcher
import cv2
import numpy as np
from ImageStitcher import ImageStitcher
import time
import os

directory = "/home/gtatum/mfly_cv/mfly-auton-cv-mapping-exploration/images"

# TODO: Use opencv for realtime image stitching
if __name__ == "__main__":
    stitcher = ImageStitcher()

    for image in os.listdir(directory):
        # check if the image ends with jpg
        if (image.endswith(".jpg")):
            img = cv2.imread(directory + "/" + image)
            stitcher.add_image(img)

    map = stitcher.get_map()
    cv2.imwrite(os.path.join(directory, 'aerial_map.png'), map)
