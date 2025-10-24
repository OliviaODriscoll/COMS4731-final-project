import cv2
import numpy as np
import os
import random
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

# --- Read the specific frame ---
frame_path = "/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/barre_tripod/barre_tripod_frame_02130.png"
frame_path = "/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/barre_tripod/barre_tripod_frame_03600.png"
print(f"Processing frame: {frame_path}")

# Read the frame
frame = cv2.imread(frame_path)

if frame is None:
    raise ValueError(f"Could not read frame from {frame_path}.")

# --- Convert to HSV for better color segmentation ---
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# --- Define tighter ranges for yellow color (shirt) ---
# More restrictive yellow ranges to avoid wall details
lower_yellow1 = np.array([22, 80, 100])   # Bright yellow, higher saturation
upper_yellow1 = np.array([28, 255, 255])

lower_yellow2 = np.array([20, 60, 80])    # Medium yellow
upper_yellow2 = np.array([30, 255, 255])

# --- Create yellow masks ---
yellow_mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
yellow_mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)

# --- Combine yellow masks ---
yellow_mask = cv2.bitwise_or(yellow_mask1, yellow_mask2)

# --- Find the center of yellow regions to focus detection ---
yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if yellow_contours:
    # Find the largest yellow contour (likely the shirt)
    largest_yellow_contour = max(yellow_contours, key=cv2.contourArea)
    # Get bounding box of the yellow region
    x, y, w, h = cv2.boundingRect(largest_yellow_contour)
    # Expand the bounding box by 150% to include nearby areas (feet, head, etc.)
    # Extra expansion downward for legs
    expand_factor_x = 1.0
    expand_factor_y = 1.5  # More expansion vertically to catch legs
    x_expand = int(w * expand_factor_x)
    y_expand = int(h * expand_factor_y)
    x_start = max(0, x - x_expand)
    y_start = max(0, y - y_expand)
    x_end = min(frame.shape[1], x + w + x_expand)
    y_end = min(frame.shape[0], y + h + y_expand)
    
    # Create a focus mask for the area around the yellow shirt
    focus_mask = np.zeros_like(yellow_mask)
    focus_mask[y_start:y_end, x_start:x_end] = 255
    print(f"Focus area: ({x_start}, {y_start}) to ({x_end}, {y_end})")
    print(f"Focus area size: {x_end - x_start} x {y_end - y_start}")
    print(f"Image size: {frame.shape[1]} x {frame.shape[0]}")
    print(f"Focus covers {((x_end - x_start) * (y_end - y_start)) / (frame.shape[1] * frame.shape[0]) * 100:.1f}% of image")
else:
    # If no yellow found, use the whole image
    focus_mask = np.ones_like(yellow_mask) * 255
    print("No yellow regions found, using full image")

# --- Multiple approaches for white detection ---
# Method 1: HSV-based detection (more inclusive)
lower_white1 = np.array([0, 0, 120])     # Very inclusive - catches light grays
upper_white1 = np.array([180, 80, 255])

lower_white2 = np.array([0, 0, 100])     # Even more inclusive
upper_white2 = np.array([180, 100, 255])

# Method 2: Grayscale thresholding (often better for white on dark)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, white_gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)  # Very low threshold

# Method 3: L*a*b* color space (better for white detection)
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l_channel = lab[:,:,0]
_, white_lab = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY)

# --- Create white masks ---
white_mask1 = cv2.inRange(hsv, lower_white1, upper_white1)
white_mask2 = cv2.inRange(hsv, lower_white2, upper_white2)
white_mask_hsv = cv2.bitwise_or(white_mask1, white_mask2)

# Combine all methods
white_mask = cv2.bitwise_or(white_mask_hsv, white_gray)
white_mask = cv2.bitwise_or(white_mask, white_lab)

# --- Apply focus mask to white detection ---
white_mask = cv2.bitwise_and(white_mask, focus_mask)

# --- SIFT-based barre detection using template matching ---
print("\n=== SIFT BARRE DETECTION ===")

# Load the barre template image
barre_template_path = "/Users/olivia/Documents/COMS4731/COMS4731-final-project/frames/barre_tripod/barre_tripod_frame_00300.png"
barre_template = cv2.imread(barre_template_path)

if barre_template is None:
    print("Could not load barre template, skipping SIFT detection")
    white_mask_no_barre = white_mask
    barre_mask = np.ones_like(white_mask) * 255
else:
    print("Loaded barre template successfully")
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Convert both images to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(barre_template, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors for both images
    kp_template, desc_template = sift.detectAndCompute(gray_template, None)
    kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)
    
    print(f"Template keypoints: {len(kp_template)}")
    print(f"Frame keypoints: {len(kp_frame)}")
    
    if desc_template is not None and desc_frame is not None:
        # Match descriptors using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc_template, desc_frame, k=2)
        
        # Apply Lowe's ratio test with more lenient matching
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:  # More lenient ratio test
                    good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
        
        # Create barre mask based on matched keypoints with additional filtering
        barre_mask = np.ones_like(white_mask) * 255
        height, width = frame.shape[:2]
        
        # Get matched keypoint locations in the frame
        matched_kp_frame = [kp_frame[m.trainIdx] for m in good_matches]
        
        # Filter matches to exclude areas that are likely the person
        filtered_matches = []
        for kp in matched_kp_frame:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Only exclude if the match is:
            # 1. On the sides of the image (where barre typically is)
            # 2. Not in the center where the person is
            # 3. Not in the lower half where legs are
            is_on_side = x < width * 0.3 or x > width * 0.7
            is_not_center = not (width * 0.3 < x < width * 0.7)
            is_not_lower_half = y < height * 0.6
            
            if is_on_side and is_not_center and is_not_lower_half:
                filtered_matches.append(kp)
                # Create larger exclusion zone around matched barre features
                radius = 60  # Larger radius for better coverage
                cv2.circle(barre_mask, (x, y), radius, 0, -1)  # Black out this area
                print(f"Excluding barre feature at ({x}, {y})")
            else:
                print(f"Keeping feature at ({x}, {y}) - likely person")
        
        # If we didn't find enough matches, use a more aggressive approach
        if len(filtered_matches) < 5:
            print("Not enough SIFT matches, using color-based barre detection")
            # Look for gray/wood colored areas that are horizontal
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal lines more aggressively
            edges = cv2.Canny(gray, 30, 100)
            horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                             minLineLength=80, maxLineGap=20)
            
            if horizontal_lines is not None:
                print(f"Found {len(horizontal_lines)} horizontal lines for barre detection")
                for line in horizontal_lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is roughly horizontal and on the sides
                    if abs(y2 - y1) < 30:  # Nearly horizontal
                        line_x_center = (x1 + x2) / 2
                        if line_x_center < width * 0.3 or line_x_center > width * 0.7:  # On sides
                            # Create thick exclusion zone around horizontal lines
                            cv2.line(barre_mask, (x1, y1), (x2, y2), 0, 25)  # 25 pixel thick line
                            print(f"Excluding horizontal barre line from ({x1}, {y1}) to ({x2}, {y2})")
        
        # Additional detection for horizontal trusses using edge detection
        # Convert to grayscale and detect horizontal edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal edges using Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Find horizontal lines using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                         minLineLength=100, maxLineGap=10)
        
        if horizontal_lines is not None:
            print(f"Found {len(horizontal_lines)} horizontal lines")
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal
                if abs(y2 - y1) < 20:  # Nearly horizontal
                    # Create exclusion zone around horizontal lines
                    cv2.line(barre_mask, (x1, y1), (x2, y2), 0, 15)  # 15 pixel thick line
                    print(f"Excluding horizontal line from ({x1}, {y1}) to ({x2}, {y2})")
        
        print(f"Applied barre mask with {len(filtered_matches)} filtered exclusion zones")
        
        # Apply barre mask to white detection
        white_mask_no_barre = cv2.bitwise_and(white_mask, barre_mask)
    else:
        print("Could not compute descriptors, using original white mask")
        white_mask_no_barre = white_mask
        barre_mask = np.ones_like(white_mask) * 255

# Find contours in the filtered white mask
white_contours, _ = cv2.findContours(white_mask_no_barre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(white_contours)} white contours after barre removal")
for i, contour in enumerate(white_contours):
    area = cv2.contourArea(contour)
    if area < 50:  # Skip very small areas
        continue
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    print(f"White contour {i}: area={area:.0f}, aspect_ratio={aspect_ratio:.2f}, pos=({x},{y}), size=({w},{h})")

# Use the barre-filtered white mask
white_mask = white_mask_no_barre

# --- Additional check: look for white areas in the lower half of the image ---
# This helps catch legs that might be outside the focus area
height, width = frame.shape[:2]
lower_half_mask = np.zeros_like(white_mask)
lower_half_mask[height//2:, :] = 255  # Lower half of image

# Create a mask for white areas in the lower half
white_lower_half = cv2.bitwise_and(white_mask, lower_half_mask)

# If we found white areas in the lower half, add them to the main white mask
if np.sum(white_lower_half > 0) > 100:  # If significant white areas found
    white_mask = cv2.bitwise_or(white_mask, white_lower_half)
    print("Found additional white areas in lower half of image")

# --- Define tighter ranges for skin color (arms, legs, head) ---
# More restrictive skin ranges to avoid wall details
lower_skin1 = np.array([0, 30, 50])      # Higher saturation for skin
upper_skin1 = np.array([20, 150, 255])
lower_skin2 = np.array([160, 30, 50])
upper_skin2 = np.array([180, 150, 255])

# --- Create skin masks ---
skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

# --- Apply focus mask to skin detection ---
skin_mask = cv2.bitwise_and(skin_mask, focus_mask)

# --- Combine yellow (shirt), white (pants), and skin (arms/legs/head) masks ---
clothing_mask = cv2.bitwise_or(yellow_mask, white_mask)
person_mask = cv2.bitwise_or(clothing_mask, skin_mask)

# --- Apply morphological operations to clean up the person mask ---
kernel = np.ones((5,5), np.uint8)  # Larger kernel for better connectivity
person_mask_clean = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
person_mask_clean = cv2.morphologyEx(person_mask_clean, cv2.MORPH_OPEN, kernel)

# --- Additional cleaning: remove small noise and fill holes ---
# Remove small noise
kernel_small = np.ones((3,3), np.uint8)
person_mask_clean = cv2.morphologyEx(person_mask_clean, cv2.MORPH_OPEN, kernel_small)

# Fill holes in the mask
kernel_fill = np.ones((7,7), np.uint8)
person_mask_clean = cv2.morphologyEx(person_mask_clean, cv2.MORPH_CLOSE, kernel_fill)

# --- Apply Gaussian blur to reduce noise in the mask ---
person_mask_final = cv2.GaussianBlur(person_mask_clean, (3, 3), 0)

# --- Convert to grayscale for additional processing ---
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# --- Optional: Background subtraction for additional refinement ---
# Only use if the blue detection alone isn't sufficient
use_bg_subtraction = False  # Set to True if you want to combine with background subtraction

if use_bg_subtraction:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    fg_mask = bg_subtractor.apply(frame)
    kernel_bg = np.ones((5,5), np.uint8)
    fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_bg)
    fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_OPEN, kernel_bg)
else:
    fg_mask_clean = np.ones_like(person_mask_final) * 255  # Use all pixels

# --- Apply Gaussian blur to reduce noise ---
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Apply bilateral filter for better edge preservation while reducing noise ---
bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)

# --- Combine color-based person detection with background subtraction ---
# This gives us the best of both approaches
combined_mask = cv2.bitwise_and(person_mask_final, fg_mask_clean)

# --- Additional refinement: remove very small regions ---
# Find contours and filter out small ones
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 1000  # Minimum area for a person (adjust based on your image size)
refined_mask = np.zeros_like(combined_mask)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        cv2.fillPoly(refined_mask, [contour], 255)

# --- Use the refined mask to isolate people ---
people_only = cv2.bitwise_and(bilateral, bilateral, mask=refined_mask)

# --- Apply edge detection to better define body boundaries ---
# Canny edge detection on the people regions
edges = cv2.Canny(people_only, 50, 150)

# --- Dilate the edges to make them thicker and more connected ---
kernel_edge = np.ones((3,3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)

# --- Combine the original mask with edge information ---
# This creates a more defined body outline
body_with_edges = cv2.bitwise_or(refined_mask, edges_dilated)

# --- Apply additional morphological operations to clean up the combined result ---
kernel_clean = np.ones((3,3), np.uint8)
body_clean = cv2.morphologyEx(body_with_edges, cv2.MORPH_CLOSE, kernel_clean)
body_clean = cv2.morphologyEx(body_clean, cv2.MORPH_OPEN, kernel_clean)

# --- Use the edge-enhanced mask for final processing ---
people_enhanced = cv2.bitwise_and(bilateral, bilateral, mask=body_clean)

# --- Improved thresholding with adaptive methods ---
# Try multiple thresholding approaches and pick the best one
_, thresh_otsu = cv2.threshold(people_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh_fixed = cv2.threshold(people_enhanced, 50, 255, cv2.THRESH_BINARY)

# Test different adaptive threshold parameters for optimal body outline
thresh_adaptive_1 = cv2.adaptiveThreshold(people_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh_adaptive_2 = cv2.adaptiveThreshold(people_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
thresh_adaptive_3 = cv2.adaptiveThreshold(people_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
thresh_adaptive_4 = cv2.adaptiveThreshold(people_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Use the first adaptive thresholding (original parameters) as it provides good body outline
thresh_adaptive = thresh_adaptive_1

# Optional: Try a hybrid approach combining Otsu and adaptive thresholding
# This can help capture both global structure (Otsu) and local details (adaptive)
hybrid_thresh = cv2.bitwise_and(thresh_otsu, thresh_adaptive)

# Choose the method that works best for your specific case
# thresh = thresh_adaptive  # Pure adaptive (good for body outline)
thresh = thresh_otsu      # Pure Otsu (good for global structure)
# thresh = thresh_adaptive  # Using adaptive as it gives good body outline

# --- Enhanced preprocessing before skeletonization ---
# Fill small holes in the binary image with multiple kernel sizes
kernel_fill_small = np.ones((3,3), np.uint8)
kernel_fill_medium = np.ones((5,5), np.uint8)
kernel_fill_large = np.ones((7,7), np.uint8)

# Progressive hole filling
thresh_filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_fill_small)
thresh_filled = cv2.morphologyEx(thresh_filled, cv2.MORPH_CLOSE, kernel_fill_medium)
thresh_filled = cv2.morphologyEx(thresh_filled, cv2.MORPH_CLOSE, kernel_fill_large)

# Remove small noise with multiple passes
kernel_noise_small = np.ones((2,2), np.uint8)
kernel_noise_medium = np.ones((3,3), np.uint8)
thresh_clean = cv2.morphologyEx(thresh_filled, cv2.MORPH_OPEN, kernel_noise_small)
thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kernel_noise_medium)

# Additional smoothing to reduce jagged edges
thresh_clean = cv2.medianBlur(thresh_clean, 3)

# --- Convert to boolean for skeletonization ---
binary = thresh_clean > 0

# --- Enhanced skeletonization with better preprocessing ---
# Apply distance transform to improve skeleton quality
dist_transform = cv2.distanceTransform(thresh_clean, cv2.DIST_L2, 5)
# Normalize the distance transform
dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
# Create a binary mask from distance transform
_, dist_binary = cv2.threshold(dist_transform, 0.3, 1.0, cv2.THRESH_BINARY)
dist_binary = (dist_binary * 255).astype(np.uint8)

# Skeletonize the distance-transformed image for better quality
skeleton = skeletonize(dist_binary > 0)

# --- Enhanced post-processing for skeleton cleanup ---
# Convert to uint8 for morphological operations
skeleton_uint8 = img_as_ubyte(skeleton)

# Remove very small skeleton branches with multiple kernel sizes
kernel_clean_1 = np.ones((2,2), np.uint8)
kernel_clean_2 = np.ones((3,3), np.uint8)
skeleton_clean = cv2.morphologyEx(skeleton_uint8, cv2.MORPH_OPEN, kernel_clean_1)
skeleton_clean = cv2.morphologyEx(skeleton_clean, cv2.MORPH_OPEN, kernel_clean_2)

# Remove isolated pixels and short branches
kernel_isolated = np.ones((3,3), np.uint8)
skeleton_final = cv2.morphologyEx(skeleton_clean, cv2.MORPH_OPEN, kernel_isolated)

# Additional cleanup: remove very short skeleton segments
# This helps remove noise while preserving important structure
kernel_short = np.ones((2,2), np.uint8)
skeleton_final = cv2.morphologyEx(skeleton_final, cv2.MORPH_OPEN, kernel_short)

# Convert back to boolean for final processing
skeleton = skeleton_final > 0

# --- Convert back to uint8 image for OpenCV display ---
skeleton_uint8 = img_as_ubyte(skeleton)

# --- Save the skeletonized result ---
output_path = "/Users/olivia/Documents/COMS4731/COMS4731-final-project/barre_tripod_frame_02130_skeletonized.png"
cv2.imwrite(output_path, skeleton_uint8)
print(f"Skeletonized image saved to: {output_path}")

# --- Save all visualization windows as JPG files ---
print("\n=== SAVING VISUALIZATION IMAGES ===")
base_output_dir = "/Users/olivia/Documents/COMS4731/COMS4731-final-project/visualizations"

# Create output directory
os.makedirs(base_output_dir, exist_ok=True)

# Save all the visualization images
visualizations = {
    "01_original": frame,
    "02_focus_mask": focus_mask,
    "03_yellow_mask_1": yellow_mask1,
    "04_yellow_mask_2": yellow_mask2,
    "05_yellow_combined": yellow_mask,
    "06_white_hsv_1": white_mask1,
    "07_white_hsv_2": white_mask2,
    "08_white_hsv_combined": white_mask_hsv,
    "09_white_grayscale": white_gray,
    "10_white_lab": white_lab,
    "11_white_before_filter": cv2.bitwise_and(cv2.bitwise_or(white_mask_hsv, cv2.bitwise_or(white_gray, white_lab)), focus_mask),
    "12_barre_template": barre_template if 'barre_template' in locals() else np.zeros_like(frame),
    "13_barre_mask": barre_mask if 'barre_mask' in locals() else np.ones_like(frame) * 255,
    "14_horizontal_lines": edges if 'edges' in locals() else np.zeros_like(frame),
    "15_white_after_sift": white_mask,
    "16_white_lower_half": white_lower_half if 'white_lower_half' in locals() else np.zeros_like(frame),
    "17_clothing_mask": clothing_mask,
    "18_skin_mask_1": skin_mask1,
    "19_skin_mask_2": skin_mask2,
    "20_skin_focused": skin_mask,
    "21_person_mask": person_mask,
    "22_person_mask_clean": person_mask_final,
    "23_refined_mask": refined_mask,
    "24_people_only": people_only,
    "25_edges": edges if 'edges' in locals() else np.zeros_like(frame),
    "26_body_with_edges": body_with_edges,
    "27_body_clean": body_clean,
    "28_people_enhanced": people_enhanced,
    "29_threshold_otsu": thresh_otsu,
    "30_threshold_fixed": thresh_fixed,
    "31_threshold_adaptive": thresh_adaptive,
    "31a_threshold_adaptive_2": thresh_adaptive_2,
    "31b_threshold_adaptive_3": thresh_adaptive_3,
    "31c_threshold_adaptive_4": thresh_adaptive_4,
    "31d_threshold_hybrid": hybrid_thresh,
    "32_threshold_filled": thresh_filled,
    "33_threshold_clean": thresh_clean,
    "33a_distance_transform": (dist_transform * 255).astype(np.uint8),
    "33b_distance_binary": dist_binary,
    "34_skeleton_raw": img_as_ubyte(skeletonize(binary)),
    "35_skeleton_clean": skeleton_clean,
    "36_skeleton_final": skeleton_uint8
}

for name, img in visualizations.items():
    if img is not None:
        output_file = os.path.join(base_output_dir, f"{name}.jpg")
        cv2.imwrite(output_file, img)
        print(f"Saved: {output_file}")

print(f"\nAll visualization images saved to: {base_output_dir}")

# --- Show results (optional - comment out if running headless) ---
cv2.imshow("Original", frame)
cv2.imshow("Focus Mask", focus_mask)
cv2.imshow("Yellow Mask 1", yellow_mask1)
cv2.imshow("Yellow Mask 2", yellow_mask2)
cv2.imshow("Combined Yellow Mask", yellow_mask)
cv2.imshow("White HSV 1", white_mask1)
cv2.imshow("White HSV 2", white_mask2)
cv2.imshow("White HSV Combined", white_mask_hsv)
cv2.imshow("White Grayscale", white_gray)
cv2.imshow("White LAB", white_lab)
cv2.imshow("White Before Filter", cv2.bitwise_and(cv2.bitwise_or(white_mask_hsv, cv2.bitwise_or(white_gray, white_lab)), focus_mask))
cv2.imshow("Barre Template", barre_template)
cv2.imshow("Barre Mask", barre_mask)
cv2.imshow("Horizontal Lines", edges)
cv2.imshow("White After SIFT", white_mask)
cv2.imshow("White Lower Half", white_lower_half)
cv2.imshow("Clothing Mask", clothing_mask)
cv2.imshow("Skin Mask 1", skin_mask1)
cv2.imshow("Skin Mask 2", skin_mask2)
cv2.imshow("Focused Skin Mask", skin_mask)
cv2.imshow("Person Mask", person_mask)
cv2.imshow("Person Mask Clean", person_mask_final)
cv2.imshow("Refined Mask", refined_mask)
cv2.imshow("People Only", people_only)
cv2.imshow("Edges", edges)
cv2.imshow("Body with Edges", body_with_edges)
cv2.imshow("Body Clean", body_clean)
cv2.imshow("People Enhanced", people_enhanced)
cv2.imshow("Threshold Otsu", thresh_otsu)
cv2.imshow("Threshold Fixed", thresh_fixed)
cv2.imshow("Threshold Adaptive", thresh_adaptive)
cv2.imshow("Threshold Filled", thresh_filled)
cv2.imshow("Threshold Clean", thresh_clean)
cv2.imshow("Skeleton Raw", img_as_ubyte(skeletonize(binary)))
cv2.imshow("Skeleton Clean", skeleton_clean)
cv2.imshow("Skeleton Final", skeleton_uint8)

# --- Print some debug info ---
print(f"Focus mask pixels: {np.sum(focus_mask > 0)}")
print(f"Yellow mask 1 pixels: {np.sum(yellow_mask1 > 0)}")
print(f"Yellow mask 2 pixels: {np.sum(yellow_mask2 > 0)}")
print(f"Combined yellow mask pixels: {np.sum(yellow_mask > 0)}")
print(f"White mask 1 pixels: {np.sum(white_mask1 > 0)}")
print(f"White mask 2 pixels: {np.sum(white_mask2 > 0)}")
print(f"Focused white mask pixels: {np.sum(white_mask > 0)}")
print(f"Clothing mask pixels: {np.sum(clothing_mask > 0)}")
print(f"Skin mask 1 pixels: {np.sum(skin_mask1 > 0)}")
print(f"Skin mask 2 pixels: {np.sum(skin_mask2 > 0)}")
print(f"Focused skin mask pixels: {np.sum(skin_mask > 0)}")
print(f"Person mask pixels: {np.sum(person_mask > 0)}")
print(f"Person mask clean pixels: {np.sum(person_mask_final > 0)}")
print(f"Refined mask pixels: {np.sum(refined_mask > 0)}")
print(f"Edge pixels found: {np.sum(edges > 0)}")
print(f"Body clean pixels: {np.sum(body_clean > 0)}")
print(f"Number of contours found: {len(contours)}")
print(f"Contours with area > {min_area}: {sum(1 for c in contours if cv2.contourArea(c) > min_area)}")
print(f"Total image pixels: {frame.shape[0] * frame.shape[1]}")
print(f"Focus area percentage: {np.sum(focus_mask > 0) / (frame.shape[0] * frame.shape[1]) * 100:.2f}%")
print(f"Final body percentage: {np.sum(body_clean > 0) / (frame.shape[0] * frame.shape[1]) * 100:.2f}%")

cv2.waitKey(0)
cv2.destroyAllWindows()
