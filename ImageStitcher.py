# File for stitching images together to form the aerial map
# Should include SIFT, kNN, RANSAC code
# Feel free to implement any helper functions you think are necessary (or don't, most of this could be one funtion tbh)
import cv2 #type: ignore
import numpy as np
from RANSAC import run_RANSAC

class ImageStitcher:
    """
    Class to hold the image stitcher for stitching the image and putting them together
    using SIFT, kNN, and RANSAC
    """
    
    # TODO: Implement - Everyone, declare what you need
    def __init__(self):
        """Initialize ImageStitcher class with default values. Contains a composite image (the map)"""
        self.__aerial_map = None
        self.__sift = cv2.SIFT_Create()
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
        """Preprocess the image for stitching: crop, grayscale, blur, histogram equalization"""
        # Crop bottom 10% to remove drone legs or props
        h = img.shape[0]
        cropped = img[:int(h * 0.9), :]

        # Convert to grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Histogram equalization to improve contrast
        equalized = cv2.equalizeHist(blurred)

        return equalized

    # TODO: Implement - Gareth
    def __run_SIFT(self, img):
        """Runs the SIFT algorithm and returns detected features"""
        keypoints = self.__sift.detect(img, None)
        keypoints, descriptors = self.__sift.compute(img, keypoints)

        return keypoints, descriptors

    # TODO: Implement - John
    def __run_kNN(self, keypoints1, descriptors1, keypoints2, descriptors2):
    
    # Initialize Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Find the 2 best matches for each feature
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe’s ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 4:
        return None, None, []

    # Extract matching keypoint coordinates
    self.baseImagePoints = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    self.newImagePoints = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return base_points, new_points, good_matches


    # TODO: Implement - Raiana
    def __run_RANSAC(self, img):
        warpedImage, H, inliers = run_RANSAC(self.baseImagePoints, self.newImagePoints, img)
        pass
