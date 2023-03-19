from PIL import Image
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
ret, img = cap.read()
#display frame in window
cv2.imshow("frame", img)
cv2.waitKey(0)

#img = Image.open("frame.jpg")
img = np.array(img)
print(img.shape)

image = cv2.Canny(img, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)
image.save("canny.png")