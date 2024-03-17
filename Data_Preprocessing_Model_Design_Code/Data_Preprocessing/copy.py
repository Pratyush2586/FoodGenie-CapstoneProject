import os
import shutil
import numpy as np

base_path = 'C:/Users/ROHIT KADAM/Desktop/Recipe1M/recipe1M_images_test/test'

output_folder = 'E:/Masters/2024/498/Recipe1M/Images/test_images'

# Ensure the output_folder exists, or create it
os.makedirs(output_folder, exist_ok=True)

count = 0

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".jpg"):  # Assuming the images have a ".jpg" extension
            img_path = os.path.join(root, file)
            print(f'Moving {img_path} to {output_folder}')
            shutil.move(img_path, output_folder)
            
            count += 1
            print(count)

print(f'Moved {count} images to {output_folder}')
