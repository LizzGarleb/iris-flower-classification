import os
import cv2
import imghdr

"""
Script for validating and cleaning image files in a directory.

This script iterates through a directory containing subdirectories of image classes, 
validates each image file, and removes invalid files. It checks for valid image extensions
and uses OpenCV (cv2) and imghdr to validate image files.

Dependencies:
    - OpenCV (imported as cv2)
    - imghdr module

Usage:
    - Specify the directory containing the image data in the 'data_dir' variable.
    - Run the script to validate and clean the image files.

Example:
    python clean_images.py

Note:
    - This script assumes that the directory structure follows a class-wise organization, 
      where each subdirectory represents a class.
    - It checks for common image file extensions (jpg, jpeg, png, bmp) and removes files 
      with invalid extensions.
    - If an image file cannot be read or processed, it will be printed as an error, but the script
      will continue processing other files.
"""

data_dir = 'data/images'  # Replace with your actual directory
image_exts = ['jpg', 'jpeg', 'png', 'bmp']

for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if os.path.isdir(class_path):
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f'Invalid file type: {image_path}')
                    os.remove(image_path)
            except Exception as e:
                print(f'Error processing file {image_path}: {e}')
