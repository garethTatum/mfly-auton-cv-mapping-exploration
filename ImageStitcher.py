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

        # Innit for kNN
        """"""
        # FLANN KD-tree for float descriptors
        index_params  = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Image B that we are comparing with. These are necessary to have
        # _ref_kp: keypoints from image B
        # _ref_desc: descriptors from image B
        self._ref_kp   = None
        self._ref_desc = None
        """"""


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
        pass

    # TODO: Implement - John
    def __run_kNN(self, img, ratio=0.75, mutual_check=True, max_keep=None):

        # Keypoint list and Descriptors array(N * 128) for the QUERY image
        kp_q, desc_q = self.__run_SIFT(img)
        # If empty, return
        if desc_q is None or len(desc_q) == 0:
            return kp_q, desc_q, []

        # kNN (k=2) in the forward direction: query -> reference
        raw = self._flann.knnMatch(desc_q, self._ref_desc, k=2)

        # Lowe's ratio test
        # Only proceed if the best match's distance is closer(less) than 0.75 of second best's match
        ratio_keep = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = raw[0], raw[1]
            if m.distance < ratio * n.distance:
                ratio_keep.append(m)
        # Mutual check
        # 4) optional mutual (cross) check: reference -> query must agree
        if mutual_check and len(ratio_keep) > 0:
            # Build a tiny reverse matcher (brute force is fine here)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            raw_rev = bf.knnMatch(self._ref_desc, desc_q, k=2)
            back_best = {}
            for pair in raw_rev:
                if len(pair) < 2:
                    continue
                m, n = pair[0], pair[1]
                if m.distance < ratio * n.distance:
                    # best ref->query for this ref descriptor
                    # (queryIdx is index in ref_desc; trainIdx is index in desc_q)
                    back_best[m.queryIdx] = m.trainIdx

            good = [m for m in ratio_keep if back_best.get(m.trainIdx, -1) == m.queryIdx]
        else:
            good = ratio_keep


        return kp_q, desc_q, good


        """Runs a kNN to match descriptors from SIFT"""


    # TODO: Implement - Raiana
    def __run_RANSAC(self, img):
        """Runs RANSAC to estimate homeography and warp image"""
        pass
