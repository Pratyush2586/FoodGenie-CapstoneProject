import os
import shutil

base_folder = r'E:\Masters\2024\498\Feature2\Feature_Ingridents'
train_folder = os.path.join(base_folder, 'train')
val_folder = os.path.join(base_folder, 'val')

# Get a list of subdirectories (classes) in the train folder
classes = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]

# Iterate through each class and copy images from val folder to the train folder
for class_name in classes:
    train_class_path = os.path.join(train_folder, class_name)
    val_class_path = os.path.join(val_folder, class_name)

    # Check if the class folder exists in the val folder
    if os.path.exists(val_class_path):
        # Copy images from val folder to train folder
        val_images = [f for f in os.listdir(val_class_path) if os.path.isfile(os.path.join(val_class_path, f))]
        for image in val_images:
            shutil.copy(os.path.join(val_class_path, image), os.path.join(train_class_path, image))
    else:
        print(f"Class '{class_name}' not found in val folder.")

print("Image combination completed.")
