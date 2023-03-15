from PIL import Image
import numpy as np
import cv2



img = Image.open("frame.jpg")
img = np.array(img)
print(img.shape)

image = cv2.Canny(img, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)
image.save("canny.png")