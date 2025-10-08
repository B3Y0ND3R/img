import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("input.jpg", 0)

plt.imshow(img, cmap="gray")
plt.title("Input Image")
plt.axis("off")
plt.show()


f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')
plt.show()


def butterworth_notch_reject(shape, d0, n, centers):
    P, Q = shape
    H = np.ones((P, Q), dtype=np.float32)
    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    centerU, centerV = Q // 2, P // 2

    for (uk, vk) in centers:

        Dk = np.sqrt((U - centerU - (uk - centerU))**2 + (V - centerV - (vk - centerV))**2)
        Dk_neg = np.sqrt((U - centerU + (uk - centerU))**2 + (V - centerV + (vk - centerV))**2)

        H *= (1 / (1 + (d0 / (Dk + 1e-6))**(2*n))) * (1 / (1 + (d0 / (Dk_neg + 1e-6))**(2*n)))

    return H


notch_centers = [(272, 256), (262, 261)] 
d0 = 5   
n = 2   

H = butterworth_notch_reject(img.shape, d0, n, notch_centers)

plt.imshow(H, cmap='gray')
plt.title("Butterworth Notch Reject Filter")
plt.axis('off')
plt.show()


G = fshift * H
magnitude_filtered = np.log(np.abs(G) + 1)
plt.imshow(magnitude_filtered, cmap='gray')
plt.title("Filtered Spectrum")
plt.axis('off')
plt.show()


G_ishift = np.fft.ifftshift(G)
img_filtered = np.fft.ifft2(G_ishift)
img_filtered = np.abs(img_filtered)

plt.imshow(img_filtered, cmap='gray')
plt.title("Filtered Image")
plt.axis('off')
plt.show()


diff = np.abs(img.astype(float) - img_filtered)
plt.imshow(diff, cmap='gray')
plt.title("Difference Image")
plt.axis('off')
plt.show()
