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
    
    # Define the scaling factor used in preprocessing
    SCALE_FACTOR = 0.4
    
    # TODO: Implement - Everyone, declare what you need
    def __init__(self):
        """Initialize ImageStitcher class with default values. Contains a composite image (the map)"""
        self.__aerial_map = None
        # self.__sift = cv2.SIFT_create()
        self.__detector = cv2.AKAZE_create(threshold=0.0005)
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
            kpB, desB = self.__detect_features(self.__aerial_map) # Note: aerial_map might be huge, consider resizing it too if this is slow
            kpN, desN = self.__detect_features(processed_img)

            newPoints, basePoints, _ = self.__run_kNN(kpN, desN, kpB, desB)
            
            # FIX 1: Scale points back up to original size
            if basePoints is not None:
                # basePoints = basePoints * (1.0 / self.SCALE_FACTOR)
                newPoints = newPoints * (1.0 / self.SCALE_FACTOR)

            # H = self.__run_RANSAC(basePoints, newPoints, processed_img)
            H = self.__compute_homography_magsac(newPoints, basePoints)

            if H is None:
                print("[WARNING] Could not find homography for add_image")
                return

            # Create canvas for Aerial Map
            h1, w1 = self.__aerial_map.shape[:2]
            
            # FIX 2: Use ORIGINAL image dimensions, not processed_img
            h2, w2 = img.shape[:2] 
            
            corners2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
            warped_corners = cv2.perspectiveTransform(corners2, H)
            all_corners = np.concatenate((np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2),
                                        warped_corners), axis=0)
            [xmin, ymin] = np.int32(np.floor(all_corners.min(axis=0).ravel()))
            [xmax, ymax] = np.int32(np.ceil(all_corners.max(axis=0).ravel()))

            trans = [-xmin, -ymin]
            H_trans = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]])
            
            # Perform image warp
            warp_width = xmax - xmin
            warp_height = ymax - ymin

            warped_img = cv2.warpPerspective(img, H_trans.dot(H), (warp_width, warp_height))
            
            shifted_base = np.zeros((warp_height, warp_width, 3), dtype=self.__aerial_map.dtype)
            shifted_base[trans[1]:h1+trans[1], trans[0]:w1+trans[0]] = self.__aerial_map

            # --- START FIX ---
            # Create a "white" image of the same size as the input frame to determine geometry
            white_img = np.ones((h2, w2), dtype=np.uint8) * 255

            # Warp the white image using the exact same transform
            mask_new = cv2.warpPerspective(white_img, H_trans.dot(H), (warp_width, warp_height))
            
            # Threshold the warped white image (handles soft edges from interpolation)
            _, mask_new = cv2.threshold(mask_new, 1, 255, cv2.THRESH_BINARY)
            
            # --- CHANGE 1: Erode mask to remove the "Crease" (black edge artifacts) ---
            kernel = np.ones((3, 3), np.uint8)
            mask_new = cv2.erode(mask_new, kernel, iterations=1)
            # --------------------------------------------------------------------------

            # --- CHANGE 2: Calculate levels dynamically based on size ---
            min_dim = min(warp_width, warp_height)
            # Ensure we don't go deeper than the image size allows (down to ~16px)
            max_possible_levels = int(np.log2(min_dim)) - 4 
            dynamic_levels = max(1, min(4, max_possible_levels))
            # -----------------------------------------------------------
            
            # Use Laplacian Blending
            self.__aerial_map = self.__laplacian_blend(shifted_base, warped_img, mask_new)

    # Code for cropping bounding box before laplacian blend to reduce memory usage
    def __overlap_bbox(self, mask):
        """
        mask: uint8 binary mask (0 or 255)
        returns: (x0, y0, x1, y1) inclusive-exclusive
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        return x0, y0, x1, y1
    
    def __laplacian_blend_roi(self, base, new, mask, levels=2, pad=16):
        """
        base, new: uint8 BGR images (same size)
        mask: uint8 binary mask (255 = use new)
        pad: extra pixels around overlap for smoothness
        """

        bbox = self.__overlap_bbox(mask)
        if bbox is None:
            return base

        x0, y0, x1, y1 = bbox

        # Expand bbox slightly (avoid hard edges)
        h, w = mask.shape
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)

        # Extract ROI
        base_roi = base[y0:y1, x0:x1]
        new_roi  = new[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]

        # Convert once to float32
        base_roi = base_roi.astype(np.float32)
        new_roi  = new_roi.astype(np.float32)
        mask_roi = (mask_roi.astype(np.float32) / 255.0)

        # Blend only ROI
        blended_roi = self.__laplacian_blend(
            base_roi, new_roi, mask_roi, levels=levels
        )

        # Paste back
        result = base.copy()
        result[y0:y1, x0:x1] = np.clip(blended_roi, 0, 255).astype(np.uint8)

        return result

    def stitch_images(self, imgs):
        keypoints = []
        descriptors = []

        # 1. Feature Detection
        for img in imgs:
            processed_img = self.__process_image(img)
            kp, desc = self.__detect_features(processed_img)
            keypoints.append(kp)
            descriptors.append(desc)

        pairwise_H = {}
        print("[INFO] Computing Homographies")
        
        # 2. Compute Pairwise Homographies
        for i in range(len(imgs)-1):
            print(f"[INFO] Matching Image {i} to {i+1}")
            basepts, newpts, matches = self.__run_kNN(keypoints[i], descriptors[i], keypoints[i+1], descriptors[i+1])
            
            # FIX 3: Pass the scale ratio to compute_homography
            # H = self.__compute_homography(keypoints[i], keypoints[i+1], matches, ratio=(1.0/self.SCALE_FACTOR))
            H = None # Default to None
            
            if basepts is not None and len(basepts) >= 4:
                ratio = 1.0 / self.SCALE_FACTOR
                src_pts = basepts * ratio
                dst_pts = newpts * ratio
                H = self.__compute_homography_magsac(src_pts, dst_pts)
                
            if H is not None:
                # Store Inverse to map backwards to the first image
                pairwise_H[(i, i + 1)] = np.linalg.inv(H)
            else:
                pairwise_H[(i, i + 1)] = np.eye(3)
            
        # 3. Build global homography chain
        print("[INFO] Building global transforms...")
        global_H = {0: np.eye(3)}
        for i in range(1, len(imgs)):
            global_H[i] = global_H[i - 1] @ pairwise_H[(i - 1, i)]

        # 4. Compute mosaic bounds
        print("[INFO] Computing mosaic canvas size...")
        all_corners = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, global_H[i])
            all_corners.append(warped)

        all_corners = np.vstack(all_corners).reshape(-1, 2)
        xmin, ymin = np.int32(all_corners.min(axis=0) - 0.5)
        xmax, ymax = np.int32(all_corners.max(axis=0) + 0.5)

        width = xmax - xmin
        height = ymax - ymin
        print(f"[INFO] Canvas size = {width} x {height}")

        # 5. Offset translation
        offsetH = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ])

        final_H = {i: offsetH @ global_H[i] for i in range(len(imgs))}

        # 6. Warping and Laplacian Blending
        print("[INFO] Warping and blending with Laplacian Pyramid...")
        
        # Start with the first image on the canvas
        mosaic = cv2.warpPerspective(imgs[0], final_H[0], (width, height))
        
        for i in range(1, len(imgs)):
            print(f"[INFO] Blending image {i+1}/{len(imgs)}...")
            
            warped_new = cv2.warpPerspective(imgs[i], final_H[i], (width, height))
            
            mask_new_gray = cv2.cvtColor(warped_new, cv2.COLOR_BGR2GRAY)
            _, mask_new = cv2.threshold(mask_new_gray, 1, 255, cv2.THRESH_BINARY)

            # --- CHANGE 1: Erode ---
            kernel = np.ones((3, 3), np.uint8)
            mask_new = cv2.erode(mask_new, kernel, iterations=1)
            
            # --- CHANGE 2: Dynamic Levels ---
            min_dim = min(width, height)
            max_possible_levels = int(np.log2(min_dim)) - 4
            dynamic_levels = max(1, min(4, max_possible_levels))

            # Binary mask where warped_new is valid
            valid_mask = (mask_new > 0)

            # Binary mask where mosaic already has content
            mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
            mosaic_valid = (mosaic_gray > 0)

            # Non-overlap: new image only
            non_overlap = valid_mask & (~mosaic_valid)

            # Paste directly
            mosaic[non_overlap] = warped_new[non_overlap]
            
            # mosaic = self.__laplacian_blend(mosaic, warped_new, mask_new, levels=2)
            mosaic = self.__laplacian_blend_roi(mosaic, warped_new, mask_new, levels=2)

        self.__aerial_map = mosaic
        print("[INFO] Stitching complete.")

    def get_map(self):
        """Returns composite aerial map"""
        return self.__aerial_map

    # =========================================================
    #  Laplacian Pyramid Helper Functions
    # =========================================================

    def __laplacian_blend(self, img1, img2, mask, levels=2):
        """
        Blends img1 and img2 using Laplacian Pyramids.
        """
        mask = mask.astype(np.float32) / 255.0
        if len(mask.shape) == 2:
            mask = cv2.merge([mask, mask, mask])

        gp1 = self.__build_gaussian_pyramid(img1, levels)
        gp2 = self.__build_gaussian_pyramid(img2, levels)
        gpM = self.__build_gaussian_pyramid(mask, levels)

        lp1 = self.__build_laplacian_pyramid(gp1)
        lp2 = self.__build_laplacian_pyramid(gp2)

        LS = []
        
        # FIX 4: Reverse gpM so it matches Laplacian order (Small -> Large)
        for l1, l2, gm in zip(lp1, lp2, gpM[::-1]):
            ls = l1 * (1.0 - gm) + l2 * gm
            LS.append(ls)

        ls_reconstruct = LS[0]
        for i in range(1, len(LS)):
            ls_reconstruct = cv2.pyrUp(ls_reconstruct)
            h, w = LS[i].shape[:2]
            ls_reconstruct = cv2.resize(ls_reconstruct, (w, h)) 
            ls_reconstruct = cv2.add(ls_reconstruct, LS[i])

        return np.clip(ls_reconstruct, 0, 255).astype(np.uint8)

    def __build_gaussian_pyramid(self, img, levels):
        gp = [img.astype(np.float32)]
        for i in range(levels):
            layer = cv2.pyrDown(gp[i])
            gp.append(layer)
        return gp

    def __build_laplacian_pyramid(self, gp):
        lp = [gp[-1]] 
        for i in range(len(gp) - 1, 0, -1):
            gaussian_expanded = cv2.pyrUp(gp[i])
            h, w = gp[i-1].shape[:2]
            gaussian_expanded = cv2.resize(gaussian_expanded, (w, h))
            
            laplacian = cv2.subtract(gp[i-1], gaussian_expanded)
            lp.append(laplacian)
        return lp


    # TODO: Implement - Daniel
    def __process_image(self, img):
        """Preprocess the image: resize (using SCALE_FACTOR), crop, grayscale, blur, equalize"""
        # Resize using the class constant
        img = cv2.resize(img, None, fx=self.SCALE_FACTOR, fy=self.SCALE_FACTOR, interpolation=cv2.INTER_AREA)
        h = img.shape[0]
        cropped = img[:int(h * 0.9), :] # Crop bottom 10%

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)

        return equalized

    # TODO: Implement - Gareth
    def __run_SIFT(self, img):
        keypoints = self.__sift.detect(img, None)
        keypoints, descriptors = self.__sift.compute(img, keypoints)
        return keypoints, descriptors

    def __detect_features(self, img):
        # AKAZE works best on grayscale, though it handles color too.
        # Ensure we are passing the processed (grayscale/equalized) image.
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        keypoints, descriptors = self.__detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    # TODO: Implement - John
    def __run_kNN(self, keypoints1, descriptors1, keypoints2, descriptors2):
        # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

        if len(good_matches) < 4:
            return None, None, []

        base_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        new_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return base_points, new_points, good_matches


    # TODO: Implement - Raiana
    def __run_RANSAC(self, baseImagePoints, newImagePoints, img):
        return run_RANSAC(baseImagePoints, newImagePoints, img)


    def __compute_homography_magsac(self, src_pts, dst_pts):
        """
        Computes Homography using USAC_MAGSAC (MAGSAC++).
        This is much more robust to outliers and elevation parallax than standard RANSAC.
        """
        if len(src_pts) < 4:
            return None
        
        # USAC_MAGSAC is built into OpenCV 4.5+
        # Threshold=5.0 is a good starting point for pixels
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
        
        # Safety Check: Discard invalid Homographies (e.g., collapsed to a line)
        if H is not None:
            # Check determinant (scale factor). If it's too tiny or huge, it's a bad warp.
            det = np.linalg.det(H[:2, :2])
            if det < 0.01 or det > 100:
                print("[WARNING] MAGSAC found a bad homography (extreme distortion). Ignoring.")
                return None
                
        return H

    def __compute_homography(self, kpts1, kpts2, matches, ratio=1.0):
        if len(matches) < 4:
            return None

        # Extract points and apply ratio scale to match original image size
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) * ratio
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) * ratio

        # Increased RANSAC threshold slightly to account for larger image scale
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H