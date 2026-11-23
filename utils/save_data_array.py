import sys
import numpy as np
import pandas as pd
from tensorflow import float32, cast
from tensorflow.io import read_file
from tensorflow.image import decode_png

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
    img = read_file(image_path)
    img = decode_png(img, channels=1)  # (28, 224, 1)    
    img_arr = cast(img, float32) / 255
    images.append(img_arr.numpy())
images = np.stack(images, axis=0)       # (60K, 28, 224, 1)

np.save(images_npy_path, images)
del images


# SAVE LABELS ARRAYS
# Save labels in the same row order as the images.
labels = df["label"].to_numpy(dtype=int)
np.save(labels_npy_path, labels)
del labels


# Reload briefly to report the final shape and confirm the cache.
cached_images = np.load(images_npy_path, mmap_mode="r")
cached_labels = np.load(labels_npy_path)

print("\nPreprocessing completed.")
print(f"  Images shape: {cached_images.shape}")
print(f"  Images dtype: {cached_images.dtype}")
print(f"  Labels shape: {cached_labels.shape}")
print(f"  Maximum label value: {np.max(cached_labels)}")
print(f"  Saved images: {images_npy_path}")
print(f"  Saved labels: {labels_npy_path}")
# RESULT:
# Preprocessing completed.
#   Images shape: (60000, 28, 224, 1)
#   Images dtype: float32
#   Labels shape: (60000,)
#   Maximum label value: 99992481
#   Saved images: ./datasets/images.npy
#   Saved labels: ./datasets/labels.npy