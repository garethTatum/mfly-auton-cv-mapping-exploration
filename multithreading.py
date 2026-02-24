
# -------------------------------------------------- NOTES --------------------------------------------------


    # So you need to do four passes

    # Each pass will have a parent thread

    # Break each parent thread into two threads that do half a pass each

    # Have a certain limiting number of threads working at the same time and make other threads wait if we are at the limit

    # At the start of each “half-pass” we start image preprocessing, keypoint detection, etc. and then when we reach the end of each “half-pass” and have taken in all the images from that half-pass, we do the homography and stitch them all together at the same time on one thread

    # And the second thread that’s already started has already started preprocessing the next half-pass set of images

# ------------------------------------------------------------------------------------------------------------

# Adapted from Raiana's Multithreading

import cv2  # type: ignore
cv2.setNumThreads(1) # add however many threads you want
import numpy as np
import threading
from RANSAC import run_RANSAC
import gc

# Thread pooling
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImageStitcher:

    SCALE_FACTOR = 0.4

    def __init__(self, max_threads=2): # change later - kept 2 so my WSL doesn't crash for testing
        self.__aerial_map = None
        self.__mosaics = []
        self.__initialized = False

        # Threading controls
        self.thread_limit = threading.Semaphore(max_threads)
        self.aerial_map_lock = threading.Lock()
        self.mosaic_lock = threading.Lock()
        
    
    # TESTING SMALL SAMPLE
    
    def run_test_small(self, images, downsample=1):
        """
        TEST MODE:
        1 pass
        2 half-passes
        2 images per half-pass (4 total)
        """
        # assert len(images) == 4

        for i in range(len(images)):
            images[i] = cv2.resize(images[i], None, fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)

        half1 = images[:len(images)//2]
        half2 = images[len(images)//2:]

        # buffer1, buffer2 = [], []
        # ready1 = threading.Event()
        # ready2 = threading.Event()

        # t1 = threading.Thread(
        #     target=self._half_pass_worker,
        #     args=(half1, buffer1, ready1)
        # )
        # t2 = threading.Thread(
        #     target=self._half_pass_worker,
        #     args=(half2, buffer2, ready2)
        # )

        # t1.start()
        # t2.start()

        # ready1.wait()
        # self._stitch_half_pass(buffer1)

        # ready2.wait()
        # self._stitch_half_pass(buffer2)

        # t1.join()
        # t2.join()

        with ThreadPoolExecutor(max_workers=self.thread_limit._value) as executor:
            futures = [
                executor.submit(self._half_pass_worker_pooling, half1),
                executor.submit(self._half_pass_worker_pooling, half2),
            ]

        for future in as_completed(futures):
            buffer = future.result()
            self._stitch_half_pass(buffer)
        
        # Stitch the two halves together
        if len(self.__mosaics) == 2:
            self.stitch_images(self.__mosaics)


    # MULTITHREADING CONTROL SYSTEM THINGY

    def run_all_passes(self, all_images):
        """
        Expects 4 passes X 36 images = 144 images total
        """
        # assert len(all_images) == 144

        passes = [
            # all_images[i * 36:(i + 1) * 36]
            all_images[i*4:(i+1)*4]
            for i in range(4)
        ]

        threads = []
        for p in passes:
            t = threading.Thread(target=self.run_pass, args=(p,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def run_pass(self, pass_images):
        """
        One pass = two half-passes of 18 images like in notes
        """
        # assert len(pass_images) == 36

        half1 = pass_images[:2]
        half2 = pass_images[2:]

        buffer1, buffer2 = [], []
        ready1 = threading.Event()
        ready2 = threading.Event()

        t1 = threading.Thread(
            target=self._half_pass_worker,
            args=(half1, buffer1, ready1)
        )
        t2 = threading.Thread(
            target=self._half_pass_worker,
            args=(half2, buffer2, ready2)
        )

        t1.start()
        t2.start()

        ready1.wait()
        self._stitch_half_pass(buffer1)

        ready2.wait()
        self._stitch_half_pass(buffer2)

        t1.join()
        t2.join()

    def _half_pass_worker(self, images, output_buffer, ready_event):
        """
        Preprocess + detect features for one half-pass
        """
        with self.thread_limit:
            for img in images:
                processed = self.__process_image(img)
                kp, desc = self.__detect_features_threadsafe(processed)
                output_buffer.append((img, processed, kp, desc))

            ready_event.set()

    def _half_pass_worker_pooling(self, images):
        """
        Preprocess + detect features for one half-pass
        """
        output_buffer = []
        with self.thread_limit:
            for img in images:
                processed = self.__process_image(img)
                kp, desc = self.__detect_features_threadsafe(processed)
                output_buffer.append((img, processed, kp, desc))

            return output_buffer

    def _stitch_half_pass(self, buffer):
        imgs = [item[0] for item in buffer]

        with self.aerial_map_lock:
            if not self.__initialized:
                self.__aerial_map = imgs[0]
                self.__initialized = True
                imgs = imgs[1:]

            if imgs:
                self.stitch_images(imgs)

    # STITCHING

    def stitch_images(self, imgs):

        keypoints = []
        descriptors = []

        for img in imgs:
            processed = self.__process_image(img)
            kp, desc = self.__detect_features_threadsafe(processed)
            keypoints.append(kp)
            descriptors.append(desc)

        pairwise_H = {}

        for i in range(len(imgs) - 1):
            basepts, newpts, _ = self.__run_kNN_threadsafe(
                keypoints[i], descriptors[i],
                keypoints[i + 1], descriptors[i + 1]
            )

            H = None
            if basepts is not None and len(basepts) >= 4:
                ratio = 1.0 / self.SCALE_FACTOR
                H = self.__compute_homography_magsac(
                    basepts * ratio, newpts * ratio
                )

            if H is not None:
                pairwise_H[(i, i + 1)] = np.linalg.inv(H)
            else:
                pairwise_H[(i, i + 1)] = np.eye(3)

        global_H = {0: np.eye(3)}
        for i in range(1, len(imgs)):
            global_H[i] = global_H[i - 1] @ pairwise_H[(i - 1, i)]

        all_corners = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, global_H[i])
            all_corners.append(warped)

        all_corners = np.vstack(all_corners).reshape(-1, 2)
        xmin, ymin = np.int32(all_corners.min(axis=0) - 0.5)
        xmax, ymax = np.int32(all_corners.max(axis=0) + 0.5)

        width = xmax - xmin
        height = ymax - ymin

        offsetH = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ])

        final_H = {i: offsetH @ global_H[i] for i in range(len(imgs))}

        mosaic = cv2.warpPerspective(imgs[0], final_H[0], (width, height))

        for i in range(1, len(imgs)):
            print(f"[INFO] Blending image {i+1}/{len(imgs)}...")
            
            warped_new = cv2.warpPerspective(imgs[i], final_H[i], (width, height))
            
            mask_new_gray = cv2.cvtColor(warped_new, cv2.COLOR_BGR2GRAY)
            _, mask_new = cv2.threshold(mask_new_gray, 1, 255, cv2.THRESH_BINARY)

            # Erode 
            kernel = np.ones((3, 3), np.uint8)
            mask_new = cv2.erode(mask_new, kernel, iterations=1)
            
            # Dynamic Levels
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
            
            # mosaic = self.__laplacian_blend_roi(mosaic, warped_new, mask_new, levels=2) # Change back to 2
            self.__laplacian_blend_roi(mosaic, warped_new, mask_new, levels=2)

            # Delete and collect data to free memory
            del warped_new, mask_new
            gc.collect()

        self.__mosaics.append(mosaic)

    # INPUT STUFF

    def __process_image(self, img):
        img = cv2.resize(img, None, fx=self.SCALE_FACTOR, fy=self.SCALE_FACTOR)
        h = img.shape[0]
        img = img[:int(h * 0.9), :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.equalizeHist(blurred)

    def __detect_features_threadsafe(self, img):
        detector = cv2.AKAZE_create(threshold=0.0005)
        return detector.detectAndCompute(img, None)

    def __run_kNN_threadsafe(self, kp1, des1, kp2, des2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.8 * n.distance]

        if len(good) < 4:
            return None, None, []

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return pts1, pts2, good

    def __compute_homography_magsac(self, src, dst):
        if len(src) < 4:
            return None

        H, _ = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0)
        if H is not None:
            det = np.linalg.det(H[:2, :2])
            if det < 0.01 or det > 100:
                return None
        return H

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
        Blends image based on detected RoI
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

        # Convert once to float32 |Changed to 16|
        base_roi = base_roi.astype(np.float16)
        new_roi  = new_roi.astype(np.float16)
        mask_roi = (mask_roi.astype(np.float16) / 255.0)

        # Blend only ROI
        blended_roi = self.__laplacian_blend(
            base_roi, new_roi, mask_roi, levels=levels
        )

        # Paste back
        base[y0:y1, x0:x1] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    def __laplacian_blend(self, img1, img2, mask, levels=2):
        mask = mask.astype(np.float32) / 255.0
        mask = cv2.merge([mask, mask, mask])

        gp1 = self.__build_gaussian_pyramid(img1, levels)
        gp2 = self.__build_gaussian_pyramid(img2, levels)
        gpM = self.__build_gaussian_pyramid(mask, levels)

        lp1 = self.__build_laplacian_pyramid(gp1)
        lp2 = self.__build_laplacian_pyramid(gp2)

        LS = []
        for l1, l2, gm in zip(lp1, lp2, gpM[::-1]):
            LS.append(l1 * (1 - gm) + l2 * gm)

        result = LS[0]
        for i in range(1, len(LS)):
            result = cv2.pyrUp(result)
            result = cv2.resize(result, (LS[i].shape[1], LS[i].shape[0]))
            result = cv2.add(result, LS[i])

        return np.clip(result, 0, 255).astype(np.uint8)

    def __build_gaussian_pyramid(self, img, levels):
        gp = [img.astype(np.float32)]
        for _ in range(levels):
            gp.append(cv2.pyrDown(gp[-1]))
        return gp

    def __build_laplacian_pyramid(self, gp):
        lp = [gp[-1]]
        for i in range(len(gp) - 1, 0, -1):
            up = cv2.pyrUp(gp[i])
            up = cv2.resize(up, (gp[i - 1].shape[1], gp[i - 1].shape[0]))
            lp.append(cv2.subtract(gp[i - 1], up))
        return lp

    def get_map(self):
        return self.__aerial_map

    def get_mosaics(self):
        return self.__mosaics
