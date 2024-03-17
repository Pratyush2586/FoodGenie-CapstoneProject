import os

validation_folder = r'E:\Masters\2024\498\Feature2\Feature_Ingridents\val'

# Get a list of subdirectories (classes) in the validation folder
classes = [d for d in os.listdir(validation_folder) if os.path.isdir(os.path.join(validation_folder, d))]

# Iterate through each class and rename the images
for class_name in classes:
    class_path = os.path.join(validation_folder, class_name)
    
    # Get a list of images in the class folder
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    
    # Rename each image by adding 'val_' prefix
    for image in images:
        old_path = os.path.join(class_path, image)
        new_name = 'val_' + image
        new_path = os.path.join(class_path, new_name)
        
        # Rename the image
        os.rename(old_path, new_path)

print("Image renaming completed.")
