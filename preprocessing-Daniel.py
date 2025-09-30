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
