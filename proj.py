import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:\\Users\\User\\.spyder-py3\\project\\9.png")
gray = img[:, :, 0]

plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)

plt.imshow(equalized, cmap='gray')
plt.axis("off")
plt.show()

cv2.imwrite('equalized.jpg', equalized)

def gaussian_blur(image, sigma=1.0):
    kernel_size = int(5 * sigma) | 1
    k = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)
    h, w = image.shape
    padded = np.pad(image, ((k, k), (k, k)), mode='reflect')
    out = np.zeros_like(image, dtype=np.float32)
    kernel = np.flip(kernel)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.sum(region * kernel)

    return np.uint8(out)

blurred = gaussian_blur(equalized, 1.0)

plt.imshow(blurred, cmap='gray')
plt.axis("off")
plt.show()

ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(thresh, cmap='gray')
plt.axis("off")
plt.show()

se = np.ones((3, 3), np.uint8)
m = cv2.erode(thresh, se)
m = cv2.dilate(m, se)

p = cv2.dilate(m, se)
p = cv2.erode(p, se)

plt.imshow(m, cmap='gray')
plt.axis("off")
plt.show()


plt.imshow(p, cmap='gray')
plt.axis("off")
plt.show()
