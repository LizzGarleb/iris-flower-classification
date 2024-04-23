import os
import cv2
import imghdr

"""
Preprocess and validate image files in a directory.

Imports:
    - os: Provides a way to interact with the operating system.
    - cv2: OpenCV library for image processing.
    - imghdr: Library to determine the type of an image.

Parameters:
    - data_dir (str): Directory containing the image data. Replace '<DIRECTORY-NAME>' with the actual directory path.
    - image_exts (list): List of valid image file extensions to filter images.

Steps:
    1. Iterate over each directory (representing image classes) in the specified data directory.
    2. For each directory, iterate over the image files.
    3. Read each image using OpenCV (cv2).
    4. Determine the image type using imghdr.
    5. Validate the image type against the allowed extensions.
    6. If the image type is not valid, print a message and remove the invalid file.
    7. If any errors occur during processing, print an error message.

Note:
    - The script assumes that each sub-directory in 'data_dir' represents a different image class.
    - It checks and removes any invalid image files based on the specified extensions.
"""

data_dir = '<DIRECTORY-NAME>'  # Replace with your actual directory
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
