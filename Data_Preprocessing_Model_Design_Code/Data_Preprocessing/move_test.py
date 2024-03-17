import os
import random
import shutil

def move_random_images(train_folder, test_folder, num_images=30):
    for class_name in os.listdir(train_folder):
        train_class_path = os.path.join(train_folder, class_name)
        test_class_path = os.path.join(test_folder, class_name)

        if os.path.isdir(train_class_path) and os.path.isdir(test_class_path):
            # Check if the test class has zero images
            if len(os.listdir(test_class_path)) == 0:
                # Get a list of images in the train class folder
                train_images = [f for f in os.listdir(train_class_path) if os.path.isfile(os.path.join(train_class_path, f))]

                # Randomly select num_images from the train class
                selected_images = random.sample(train_images, min(num_images, len(train_images)))

                # Move selected images to the test class folder
                for image in selected_images:
                    src_path = os.path.join(train_class_path, image)
                    dest_path = os.path.join(test_class_path, image)
                    shutil.move(src_path, dest_path)

                print(f"Moved {len(selected_images)} images from '{class_name}' in train to test.")

# Set your base and train/test folder paths
base_folder = r'E:\Masters\2024\498\Feature2\Feature_Ingridents'
train_folder = os.path.join(base_folder, 'train')
test_folder = os.path.join(base_folder, 'test')

# Call the function to move random images
move_random_images(train_folder, test_folder)
