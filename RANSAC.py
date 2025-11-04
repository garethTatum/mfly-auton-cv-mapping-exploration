import numpy as np
import cv2 # type: ignore

def extract_points_from_matches(keypoints1, keypoints2, matches):

    baseImageCoordinates = []
    newImageCoordinates = []

    for m in matches:

        if isinstance(m, cv2.DMatch):
            baseImageKeypointIndex = m.queryIdx
            newImageKeypointIndex = m.trainIdx
            
        else:
        # Skipping faulty entries where match is not a tuple
            if len(m) < 2:
                continue
            
            baseImageKeypointIndex = int(m[0])
            newImageKeypointIndex = int(m[1])

        # Skipping faulty entries out of range
        if ((baseImageKeypointIndex < 0 or baseImageKeypointIndex >= len(keypoints1)) or (newImageKeypointIndex < 0 or newImageKeypointIndex >= len(keypoints2))):
            continue

        # kp.pt is (x, y) as floats
        baseImageCoordinates.append(keypoints1[baseImageKeypointIndex].pt)
        newImageCoordinates.append(keypoints2[newImageKeypointIndex].pt)

    if len(baseImageCoordinates) == 0:
        return None, None

    baseImagePoints = np.array(baseImageCoordinates, dtype=np.float32)
    newImagePoints = np.array(newImageCoordinates, dtype=np.float32)

    return baseImagePoints, newImagePoints

def find_best_inliers_and_homography(baseImagePoints, newImagePoints, maxIterations=2000, threshold=5.0, confidence=0.99):

    # Ensure we have at least four matches
    
    numberOfMatches = baseImagePoints.shape[0]
    if numberOfMatches < 4:
        return None, None

    # Initializing variables for best model

    best_H = None
    best_inliers = None
    max_inliers = 0
    
    # Finding best model

    for element in range(maxIterations):
        
        # 1. Randomly sample 4 correspondences
        indices = np.random.choice(numberOfMatches, 4, replace=False)
        sampleBaseImagePoints = baseImagePoints[indices]
        sampleNewImagePoints = newImagePoints[indices]

        # 2. Compute candidate homography from 4 points
        
        # method = 0 is DLT, findHomography returns 
        H, mask = cv2.findHomography(sampleBaseImagePoints, sampleNewImagePoints, method=0)
        
        if H is None:
            continue

        # 3. Project all base image points using this H
        baseImagePointsHomography = np.hstack([baseImagePoints, np.ones((numberOfMatches, 1))])
                
        projected = (H @ baseImagePointsHomography.T).T
                
        projected = projected[:, 0:2] / projected[:, 2:3]
        
        # 4. Compute reprojection error
        errors = np.linalg.norm(projected - newImagePoints, axis=1)
        
        # 5. Identify inliers
        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        # 6. Update best model if more inliers found
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers = inliers

            # Early stopping if enough inliers found
            if max_inliers > numberOfMatches * confidence:
                break

    # 7. Recompute homography using all inliers (refinement step)
    if ((best_inliers is not None) and (np.sum(best_inliers) >= 4)):
        best_H, mask = cv2.findHomography(baseImagePoints[best_inliers], newImagePoints[best_inliers], method=0) # DLT

    return best_H, best_inliers


def run_RANSAC(baseImagePoints, newImagePoints, img):

    # Step 1: Run custom RANSAC to get homography and inlier mask
    H, inliers = find_best_inliers_and_homography(baseImagePoints, newImagePoints,
                             maxIterations=2000, threshold=5.0, confidence=0.99)
    
    # Step 2: Check if a valid homography was found
    if H is None:
        print("RANSAC failed: not enough points to compute homography")
        return None, None, None

    # Step 3: Warp the image using the computed homography
    height, width = img.shape[0:2]
    warpedImage = cv2.warpPerspective(img, H, (width, height))
    
    # Step 4: Return results
    return warpedImage, H, inliers