import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import os
import sys

def resize_keep_ratio(image, height=800):
    """Resize image to given height while keeping aspect ratio. Returns resized image and scaling factor (orig_h / new_h)."""
    resized = imutils.resize(image, height=height)
    scaling_factor = image.shape[0] / float(resized.shape[0])
    return resized, scaling_factor

def preprocess_for_edges(image):
    """
    Convert to grayscale, blur, compute adaptive Canny thresholds using median,
    and return gray + edged images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    m = np.median(blur)  # median of the single-channel blurred image
    sigma = 0.33
    lower_threshold = int(max(0, (1.0 - sigma) * m))
    upper_threshold = int(min(255, (1.0 + sigma) * m))
    edged = cv2.Canny(blur, lower_threshold, upper_threshold)
    return gray, edged

def find_document_contour(edged):
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)  # Gets the polygonal approx
        if len(approx) == 4:
            return approx  # returning coordinates: 4 corner points of page
    return None

def adaptive_binarize(gray):
    # Correct adaptiveThreshold call: src, maxValue, adaptiveMethod, thresholdType, blockSize, C
    return cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 11, 2)

def scan_image(image_path, output_dir="output", debug=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # --- FIX: actually copy the array (call the method) ---
    img_copy = image.copy()

    resized, scaling_factor = resize_keep_ratio(image, height=800)
    # Using resized image for detection
    gray, edged = preprocess_for_edges(resized)
    doc_cnts = find_document_contour(edged)
    
    
    base = os.path.splitext(os.path.basename(image_path))[0]
    debug_base = os.path.join(output_dir, base)

    # Save debug for intermediate images
    cv2.imwrite(f"{debug_base}_resized.jpg", resized)
    cv2.imwrite(f"{debug_base}_gray.jpg", gray)
    cv2.imwrite(f"{debug_base}_edged.jpg", edged)

    scanned_color_path = None

    if doc_cnts is not None:
        draw = resized.copy()
        cv2.drawContours(draw, [doc_cnts], -1, (0,255,0), 3)
        cv2.imwrite(f"{debug_base}_contour_check.jpg", draw)
        print("Contour drawn on resized image saved.")



    if doc_cnts is None:
        print("Document contour not found.")
        scanned_bw_img = adaptive_binarize(gray)
    else:
        # scale contour points back to original image size
        pts = doc_cnts.reshape(4, 2).astype("float32") * scaling_factor

        # Use four_point_transform on the ORIGINAL image (high-res)
        warped = four_point_transform(img_copy, pts)  # transforms trapezium -> rectangle and crops

        # save colored warp
        scanned_color_path = f"{debug_base}_scanned_color.jpg"
        cv2.imwrite(scanned_color_path, warped)

        # convert to gray and binarize
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        scanned_bw_img = adaptive_binarize(warped_gray)

        if debug:
            draw = resized.copy()
            cv2.drawContours(draw, [doc_cnts], -1, (0, 255, 0), 2)
            cv2.imwrite(f"{debug_base}_contour.jpg", draw)

    scanned_bw_path = f"{debug_base}_scanned_bw.jpg"
    cv2.imwrite(scanned_bw_path, scanned_bw_img)
    print("Saved:", scanned_bw_path)
    if scanned_color_path:
        print("Saved:", scanned_color_path)

    return {"scanned_bw": scanned_bw_path, "scanned_color": scanned_color_path}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python doc_scanner.py path/to/image.jpg [--debug]")
        sys.exit(0)
    image_path = sys.argv[1]
    debug_flag = "--debug" in sys.argv
    result = scan_image(image_path, output_dir="output", debug=debug_flag)
    print(result)
