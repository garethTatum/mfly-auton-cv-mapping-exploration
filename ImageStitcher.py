# File for stitching images together to form the aerial map
# Should include SIFT, kNN, RANSAC code
# Feel free to implement any helper functions you think are necessary (or don't, most of this could be one funtion tbh)
import cv2
import numpy as np

class ImageStitcher:
    """
    Class to hold the image stitcher for stitching the image and putting them together
    using SIFT, kNN, and RANSAC
    """
    
    # TODO: Implement - Everyone, declare what you need
    def __init__(self):
        """Initialize ImageStitcher class with default values. Contains a composite image (the map)"""
        self.__aerial_map = None
        pass

    # TODO: Implement - Gareth
    def add_image(self, img):
        """Add an image to the composite map"""
        pass

    def get_map(self):
        """Returns composite aerial map"""
        return self.__aerial_map

    # TODO: Implement - Daniel
    def __process_image(self, img):
        """Preprocess the image for stitching"""
        pass

    # TODO: Implement - Gareth
    def __run_SIFT(self, img):
        """Runs the SIFT algorithm and returns detected features"""
        pass

    # TODO: Implement - John
    def __run_kNN(self, img):
        """Runs a kNN to match descriptors from SIFT"""
        pass

    # TODO: Implement - Raiana
    def __run_RANSAC(self, img):
        """Runs RANSAC to estimate homeography and warp image"""
        pass