import os
import cv2
import imghdr

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
