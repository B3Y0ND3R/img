import cv2
import numpy as np
from google.colab.patches import cv2_imshow


img=cv2.imread("9.png")
gray=img[:, :, 0]


cv2_imshow(img)
clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized=clahe.apply(gray)

print("CLAHE Equalized Image:")
cv2_imshow(equalized)
cv2.imwrite('equalized.jpg',equalized)


def gaussian_blur(image,sigma=1.0):
    kernel_size= int(5*sigma) | 1
    k=kernel_size // 2
    kernel=np.zeros((kernel_size, kernel_size),dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            x=i-k
            y=j-k
            kernel[i,j] =np.exp(-(x**2 + y**2)/(2*sigma**2))

    sum=0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            sum+=kernel[i, j]

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j]/=sum

    h,w=image.shape
    padded=np.pad(image,((k, k),(k, k)))
    out=np.zeros_like(image, dtype=np.float32)
    kernel=np.flip(kernel)

    for i in range(h):
        for j in range(w):
            region=padded[i:i+kernel_size,j:j+kernel_size]
            out[i,j]=np.sum(region * kernel)

    return np.uint8(out)


blurred=gaussian_blur(equalized,1.0)
print("Gaussian Blurred Image:")
cv2_imshow(blurred)


ret,thresh=cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu Thresholded Image:")
cv2_imshow(thresh)


se=np.ones((3, 3), np.uint8)
#m=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se, iterations=2)
m=cv2.erode(thresh, se)
m=cv2.dilate(m,se)
print("Morphological Opening Result:")
cv2_imshow(m)