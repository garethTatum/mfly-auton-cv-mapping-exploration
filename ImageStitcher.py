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
        self.__sift = cv2.SIFT_create()
        self.__initialized = False
        pass

    # TODO: Implement - Gareth
    def add_image(self, img):
        """Add an image to the composite map"""
        if not self.__initialized:
            self.__aerial_map = img
            self.__initialized = True
        else:
            # Running through preprocessing, SIFT, kNN, and RANSAC
            processed_img = self.__process_image(img)
            kpB, desB = self.__run_SIFT(self.__aerial_map)
            kpN, desN = self.__run_SIFT(processed_img)

            basePoints, newPoints,_ = self.__run_kNN(kpB, desB, kpN, desN)

            H = self.__run_RANSAC(basePoints, newPoints, processed_img)

            # Create canvas for Aerial Map
            h1, w1 = self.__aerial_map.shape[:2]
            h2, w2 = processed_img.shape[:2]
            corners2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
            warped_corners = cv2.perspectiveTransform(corners2, H)
            all_corners = np.concatenate((np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2),
                                        warped_corners), axis=0)
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

            trans = [-xmin, -ymin]
            H_trans = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]])
            size = (xmax-xmin, ymax-ymin)

            # Perform image warp (originally in RANSAC)
            height, width = img.shape[0:2]
            warped_img = cv2.warpPerspective(img, H_trans.dot(H), (width, height))
            
            shifted_base = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=self.__aerial_map.dtype)
            shifted_base[trans[1]:h1+trans[1], trans[0]:w1+trans[0]] = self.__aerial_map

            # Blend Images
            mask_new = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            _, mask_new = cv2.threshold(mask_new, 1, 255, cv2.THRESH_BINARY)

            
            # Double-check mask matches the canvas
            if mask_new.shape[:2] != shifted_base.shape[:2]:
                mask_new = cv2.resize(mask_new, (shifted_base.shape[1], shifted_base.shape[0]))

            result = shifted_base.copy()
            result[mask_new == 255] = warped_img[mask_new == 255]

            self.__aerial_map = result

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
        baseImagePoints = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        newImagePoints = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return baseImagePoints, newImagePoints, good_matches

    # TODO: Implement - Raiana
    def __run_RANSAC(self, baseImagePoints, newImagePoints, img):
        return run_RANSAC(baseImagePoints, newImagePoints, img)

