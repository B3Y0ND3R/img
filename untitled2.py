import cv2
import numpy as np
import matplotlib.pyplot as plt

duck = cv2.imread("input.jpg", 0)

plt.imshow(duck, cmap="gray")
plt.axis("off")
plt.show()

ft = np.fft.fft2(duck)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
#magnitude_spectrum = 20 * np.log(magnitude_spectrum_ac + 1)
magnitude_spectrum = np.log(magnitude_spectrum_ac)
#magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.show()

phase = np.angle(ft_shift)
#phase_ = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.imshow(phase, cmap='gray')
plt.title("Phase Spectrum")
plt.show()

def notch_filter(img, notch_centers, radius):
    h, w = img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    notch_mask = np.ones((h, w), dtype=np.float32)
    center_u, center_v = h // 2, w // 2
    for center in notch_centers:
        u_notch, v_notch = center
        D1 = np.sqrt((u - (center_v + u_notch - center_v))**2 + (v - (center_u + v_notch - center_u))**2)
        D2 = np.sqrt((u - (center_v - (u_notch - center_v)))**2 + (v - (center_u - (v_notch - center_u)))**2)
        notch_mask = notch_mask * (D1 > radius) * (D2 > radius)

    return notch_mask

notch_centers = [(272, 256), (262, 261)]
radius = 5
notch_mask = notch_filter(duck, notch_centers, radius)

plt.imshow(notch_mask, cmap='gray')
plt.title("Notch Filter Mask")
plt.axis('off')
plt.show()

ft_filtered = ft_shift * notch_mask
magnitude_filtered = 20 * np.log(np.abs(ft_filtered) + 1)
plt.imshow(magnitude_filtered, cmap='gray')
plt.title("Filtered Spectrum")
plt.axis('off')
plt.show()

ft_ishift = np.fft.ifftshift(ft_filtered)
img_filtered = np.fft.ifft2(ft_ishift)
img_filtered = np.abs(img_filtered)

plt.imshow(img_filtered, cmap='gray')
plt.title("Filtered Image")
plt.axis('off')
plt.show()

diff_img = np.abs(duck.astype(float) - img_filtered)
plt.imshow(diff_img, cmap='gray')
plt.title("Difference Image")
plt.axis('off')
plt.show()
