import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# 1
# Load an image from the folder
image_path = "data/images/Leskovec.jpg"
image = cv2.imread(image_path)  # Read the image

# Convert BGR to RGB (because OpenCV loads in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.axis("off")  # Hide axes
plt.title("Original Image")
plt.show()

# 2
# Folder path
image_folder = "data/images"

# Get all image file names
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg", ".webp"))]

# Resize images
resized_images = []
for file in image_files:
    img = cv2.imread(os.path.join(image_folder, file))
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    resized_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB

# Display resized images
fig, axes = plt.subplots(2, len(resized_images) // 2, figsize=(12, 6))

for ax, img, name in zip(axes.flat, resized_images, image_files):
    ax.imshow(img)
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()

# 3
gray_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in resized_images]

fig, axes = plt.subplots(2, len(gray_images) // 2, figsize=(12, 6))

for ax, img, name in zip(axes.flat, gray_images, image_files):
    ax.imshow(img, cmap="gray")  # Use grayscale colormap
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()

# 4
# Select an image
image = resized_images[0]

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  # (15,15) is the kernel size

# Display side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(blurred_image)
axes[1].set_title("Blurred")
axes[1].axis("off")

plt.show()

# 5
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)  # Thresholds: 100, 200

# Display side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale")
axes[0].axis("off")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Edges Detected")
axes[1].axis("off")

plt.show()