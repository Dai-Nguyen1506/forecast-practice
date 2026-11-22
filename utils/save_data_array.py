import sys
import numpy as np
import pandas as pd
from PIL import Image

IMG_H = 28
IMG_W = 224

csv_path = "./datasets/labels.csv"
image_dir = "./datasets/images/"
images_npy_path = "./datasets/images.npy"
labels_npy_path = "./datasets/labels.npy"

df = pd.read_csv(csv_path, dtype=str) # Read labels as strings to preserve leading zeros.
images = []
labels = []

# SAVE IMAGES ARRAYS
for i, image_name in enumerate(df["image"]):
    image_path = image_dir + image_name

    # Keep the exact preprocessing used in Thu_Model.ipynb.
    with Image.open(image_path) as img:
        img = img.convert('L')
        img = img.resize((IMG_W, IMG_H))
        img_arr = np.array(img, dtype=np.float32) / 255 # chuyển ảnh thành ma trận số và scale về [0, 1]
        images.append(img_arr)
images = np.array(images, dtype=np.float32)
images = np.expand_dims(np.array(images), axis=-1) # Thêm chiều kênh (channel) để phù hợp với đầu vào của mô hình CNN.

np.save(images_npy_path, images)
del images  


# SAVE LABELS ARRAYS
# Save labels in the same row order as the images.
labels = df["label"].to_numpy(dtype=str)
np.save(labels_npy_path, labels)
del labels

# Reload briefly to report the final shape and confirm the cache.
cached_images = np.load(images_npy_path, mmap_mode="r")
cached_labels = np.load(labels_npy_path, allow_pickle=False)

print("\nPreprocessing completed.")
print(f"  Images shape: {cached_images.shape}")
print(f"  Images dtype: {cached_images.dtype}")
print(f"  Labels shape: {cached_labels.shape}")
print(f"  Maximum label length: {max(map(len, cached_labels))}")
print(f"  Saved images: {images_npy_path}")
print(f"  Saved labels: {labels_npy_path}")
# RESULT:
# Preprocessing completed.
#   Images shape: (60000, 28, 224, 1)
#   Images dtype: float32
#   Labels shape: (60000,)
#   Maximum label length: 8
#   Saved images: ./datasets/images.npy
#   Saved labels: ./datasets/labels.npy