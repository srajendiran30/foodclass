import cv2
import numpy as np

img_bgr = cv2.imread("test.jpg")
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL)

# Filter out low saturation values, which means gray-scale pixels(majorly in background)
bgd_mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([255, 30, 255]))

# Get a mask for pitch black pixel values
black_pixels_mask = cv2.inRange(img_bgr, np.array([0, 0, 0]), np.array([70, 70, 70]))

# Get the mask for extreme white pixels.
white_pixels_mask = cv2.inRange(img_bgr, np.array([230, 230, 230]), np.array([255, 255, 255]))

final_mask = cv2.max(bgd_mask, black_pixels_mask)
final_mask = cv2.min(final_mask, ~white_pixels_mask)
final_mask = ~final_mask

final_mask = cv2.erode(final_mask, np.ones((3, 3), dtype=np.uint8))
final_mask = cv2.dilate(final_mask, np.ones((5, 5), dtype=np.uint8))

# Now you can finally find contours.
contours, hierarchy = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

final_contours = []
count=0
for contour in contours:
	area = cv2.contourArea(contour)
	if area > 14000:
		print("one contour is invalid")
	else:
		final_contours.append(contour)
		count=count+1;

if (count== 0):
	print("Image is invalid")

for i in range(len(final_contours)):
	img_bgr = cv2.drawContours(img_bgr, final_contours, -1, (0,255,0), 3)

debug_img = img_bgr
debug_img = cv2.resize(debug_img, None, fx=0.3, fy=0.3)
cv2.imwrite("./out.png", debug_img)
