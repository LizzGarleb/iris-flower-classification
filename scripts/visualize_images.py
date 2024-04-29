import tensorflow as tf
from matplotlib import pyplot as plt

"""
Script to visualize a batch of images from a directory using TensorFlow and Matplotlib.

This script loads a batch of images from a directory using TensorFlow's `image_dataset_from_directory` function 
and visualizes a subset of these images using Matplotlib. It displays a grid of images with their corresponding 
labels.

Dependencies:
    - TensorFlow (imported as tf)
    - Matplotlib (imported as plt)

Usage:
    - Place the images you want to visualize in the specified directory ('data/images').
    - Run the script to visualize a batch of images.

Example:
    python visualize_images.py

Note:
    - This script assumes that images are organized in subdirectories within the specified directory, where each
      subdirectory represents a class.
    - The script visualizes the first four images in the batch. You can modify the script to visualize a different
      subset or all images.
"""

data = tf.keras.utils.image_dataset_from_directory('data/images')

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

plt.show()

# 0 = setosa 1 = versicolor 2 = virginica
