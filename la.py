import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Picture2.jpg", cv2.IMREAD_GRAYSCALE)

def glcm_horizontal(image, levels=256):
    h, w = image.shape
    glcm = np.zeros((levels, levels), dtype=np.int32)
    for y in range(h):
        for x in range(w-1):
            i = image[y, x]
            j = image[y, x+1]
            glcm[i, j] += 1
    return glcm

def glcm_vertical(image, levels=256):
    h, w = image.shape
    glcm = np.zeros((levels, levels), dtype=np.int32)
    for y in range(h-1):
        for x in range(w):
            i = image[y, x]
            j = image[y+1, x]
            glcm[i, j] += 1
    return glcm

def glcm_diagonal(image, levels=256):
    h, w = image.shape
    glcm = np.zeros((levels, levels), dtype=np.int32)
    for y in range(h-1):
        for x in range(w-1):
            i = image[y, x]
            j = image[y+1, x+1]
            glcm[i, j] += 1
    return glcm

def normalize_glcm(glcm):
    total = 0
    h, w = glcm.shape
    for i in range(h):
        for j in range(w):
            total += glcm[i, j]
    norm = np.zeros_like(glcm, dtype=float)
    for i in range(h):
        for j in range(w):
            if total > 0:
                norm[i, j] = glcm[i, j] / total
    return norm

def max_probability(glcm):
    h, w = glcm.shape
    max_val = 0
    for i in range(h):
        for j in range(w):
            if glcm[i, j] > max_val:
                max_val = glcm[i, j]
    return max_val

def energy(glcm):
    h, w = glcm.shape
    total = 0.0
    for i in range(h):
        for j in range(w):
            total += glcm[i, j] ** 2
    return total

def entropy(glcm):
    h, w = glcm.shape
    total = 0.0
    for i in range(h):
        for j in range(w):
            if glcm[i, j] > 0:
                total += -glcm[i, j] * np.log2(glcm[i, j])
    return total

def contrast(glcm):
    h, w = glcm.shape
    total = 0.0
    for i in range(h):
        for j in range(w):
            total += ((i - j) ** 2) * glcm[i, j]
    return total

def homogeneity(glcm):
    h, w = glcm.shape
    total = 0.0
    for i in range(h):
        for j in range(w):
            total += glcm[i, j] / (1 + abs(i - j))
    return total

def compute_features(glcm_raw):
    glcm = normalize_glcm(glcm_raw)
    return {
        "Max Probability": max_probability(glcm),
        "Energy": energy(glcm),
        "Entropy": entropy(glcm),
        "Contrast": contrast(glcm),
        "Homogeneity": homogeneity(glcm)
    }

glcm_h = glcm_horizontal(img)
glcm_v = glcm_vertical(img)
glcm_d = glcm_diagonal(img)

features_h = compute_features(glcm_h)
features_v = compute_features(glcm_v)
features_d = compute_features(glcm_d)

def print_features(direction, features):
    print(f"{direction}")
    for key, value in features.items():
        print(f"{key:15}:{round(value, 4)}")
    print()

print_features("Horizontal", features_h)
print_features("Vertical", features_v)
print_features("Diagonal", features_d)

plt.imshow(img, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(20,5))

plt.subplot(1,4,1)
plt.imshow(normalize_glcm(glcm_h), cmap="hot")
plt.title("Horizontal GLCM")

plt.subplot(1,4,2)
plt.imshow(normalize_glcm(glcm_v), cmap="hot")
plt.title("Vertical GLCM")

plt.subplot(1,4,3)
plt.imshow(normalize_glcm(glcm_d), cmap="hot")
plt.title("Diagonal GLCM")

plt.show()