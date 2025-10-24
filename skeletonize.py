import cv2
import numpy as np
import os
import random
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

# --- Read a random frame from frames directory ---
frames_dir = "frames"
frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]

if not frame_files:
    raise ValueError("No PNG files found in frames directory.")

# Select a random frame
random_frame = random.choice(frame_files)
frame_path = os.path.join(frames_dir, random_frame)
print(f"Selected random frame: {random_frame}")

# Read the frame
frame = cv2.imread(frame_path)

if frame is None:
    raise ValueError(f"Could not read frame from {frame_path}.")

# --- Convert to HSV for better color segmentation ---
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# --- Define multiple ranges for blue color (leotards) ---
# Try different blue ranges to catch variations in lighting and fabric
lower_blue1 = np.array([100, 50, 50])   # Brighter blue
upper_blue1 = np.array([130, 255, 255])

lower_blue2 = np.array([90, 30, 30])    # Darker blue
upper_blue2 = np.array([140, 255, 255])

lower_blue3 = np.array([110, 40, 40])   # Medium blue
upper_blue3 = np.array([125, 255, 255])

# --- Create multiple blue masks ---
blue_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
blue_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
blue_mask3 = cv2.inRange(hsv, lower_blue3, upper_blue3)

# --- Combine all blue masks ---
blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
blue_mask = cv2.bitwise_or(blue_mask, blue_mask3)

# --- Define ranges for skin color (arms, legs, head) ---
# Skin color in HSV: Hue around 0-20 and 160-180, Saturation 20-170, Value 35-255
lower_skin1 = np.array([0, 20, 35])
upper_skin1 = np.array([20, 170, 255])
lower_skin2 = np.array([160, 20, 35])
upper_skin2 = np.array([180, 170, 255])

# --- Create skin masks ---
skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

# --- Combine blue (torso) and skin (arms/legs/head) masks ---
person_mask = cv2.bitwise_or(blue_mask, skin_mask)

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

# --- Threshold the edge-enhanced image ---
_, thresh = cv2.threshold(people_enhanced, 50, 255, cv2.THRESH_BINARY)

# --- Convert to boolean for skeletonization ---
binary = thresh > 0

# --- Skeletonize ---
skeleton = skeletonize(binary)

# --- Convert back to uint8 image for OpenCV display ---
skeleton_uint8 = img_as_ubyte(skeleton)

# --- Show results ---
cv2.imshow("Original", frame)
cv2.imshow("Blue Mask", blue_mask)
cv2.imshow("Skin Mask 1", skin_mask1)
cv2.imshow("Skin Mask 2", skin_mask2)
cv2.imshow("Combined Skin Mask", skin_mask)
cv2.imshow("Person Mask", person_mask)
cv2.imshow("Person Mask Clean", person_mask_final)
cv2.imshow("Refined Mask", refined_mask)
cv2.imshow("People Only", people_only)
cv2.imshow("Edges", edges)
cv2.imshow("Body with Edges", body_with_edges)
cv2.imshow("Body Clean", body_clean)
cv2.imshow("People Enhanced", people_enhanced)
cv2.imshow("Thresholded", thresh)
cv2.imshow("Skeletonized", skeleton_uint8)

# --- Print some debug info ---
print(f"Blue mask pixels: {np.sum(blue_mask > 0)}")
print(f"Skin mask 1 pixels: {np.sum(skin_mask1 > 0)}")
print(f"Skin mask 2 pixels: {np.sum(skin_mask2 > 0)}")
print(f"Combined skin mask pixels: {np.sum(skin_mask > 0)}")
print(f"Person mask pixels: {np.sum(person_mask > 0)}")
print(f"Person mask clean pixels: {np.sum(person_mask_final > 0)}")
print(f"Refined mask pixels: {np.sum(refined_mask > 0)}")
print(f"Edge pixels found: {np.sum(edges > 0)}")
print(f"Body clean pixels: {np.sum(body_clean > 0)}")
print(f"Number of contours found: {len(contours)}")
print(f"Contours with area > {min_area}: {sum(1 for c in contours if cv2.contourArea(c) > min_area)}")
print(f"Total image pixels: {frame.shape[0] * frame.shape[1]}")
print(f"Final body percentage: {np.sum(body_clean > 0) / (frame.shape[0] * frame.shape[1]) * 100:.2f}%")

cv2.waitKey(0)
cv2.destroyAllWindows()
