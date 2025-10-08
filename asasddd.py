import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)
magnitude = np.abs(Fshift)

plt.figure(figsize=(6,6))
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


log_spectrum = np.log(1 + magnitude)
plt.figure(figsize=(6,6))
plt.title("Power Spectrum (Log Scale)", fontsize=12)
plt.imshow(log_spectrum, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

phase = np.angle(Fshift)
plt.figure(figsize=(6,6))
plt.title("Phase Spectrum", fontsize=12)
plt.imshow(phase, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

def butterworth_notch_filter(shape, uk, vk, D0k=5, n=2):
    M, N = shape
    H = np.ones((M, N), dtype=np.float64)
    
    u_center = M / 2
    v_center = N / 2
    
    for u in range(M):
        for v in range(N):
            Dk = np.sqrt((u - u_center - uk)**2 + (v - v_center - vk)**2)
            D_neg_k = np.sqrt((u - u_center + uk)**2 + (v - v_center + vk)**2)
            H[u, v] = (1 / (1 + (D0k / (Dk + 1e-6))**(2*n))) * (1 / (1 + (D0k / (D_neg_k + 1e-6))**(2*n)))
    
    return H

M, N = img.shape
uk1 = 272 - M//2 
vk1 = 256 - N//2 

H1 = butterworth_notch_filter(img.shape, uk1, vk1, D0k=5, n=2)

plt.figure(figsize=(6,6))
plt.title("Butterworth Notch Filter", fontsize=12)
plt.imshow(H1**0.3, cmap='gray')  
plt.axis('off')
plt.tight_layout()
plt.show()

F_filtered1 = Fshift * H1
F_ishift1 = np.fft.ifftshift(F_filtered1)
img_filtered1 = np.fft.ifft2(F_ishift1)
img_filtered1 = np.abs(img_filtered1)

uk2 = 262 - M//2
vk2 = 261 - N//2

H2 = butterworth_notch_filter(img.shape, uk2, vk2, D0k=5, n=2)

F_filtered2 = Fshift * H2
F_ishift2 = np.fft.ifftshift(F_filtered2)
img_filtered2 = np.fft.ifft2(F_ishift2)
img_filtered2 = np.abs(img_filtered2)

log_spectrum_filtered1 = np.log(1 + np.abs(F_filtered2))
plt.figure(figsize=(6,6))
plt.title("Filtered Power Spectrum\n(Noise frequencies removed)", fontsize=12)
plt.imshow(log_spectrum_filtered1, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.title("Filtered Image 1")
plt.imshow(img_filtered1, cmap='gray')
plt.axis('off')
plt.show()

plt.figure(figsize=(6,6))
plt.title("Filtered Image 2")
plt.imshow(img_filtered2, cmap='gray')
plt.axis('off')
plt.show()