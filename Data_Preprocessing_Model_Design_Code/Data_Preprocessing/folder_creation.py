import os
import shutil

base_folder = r'E:\Masters\2024\498\Feature2\Feature_Ingridents'
train_folder = os.path.join(base_folder, 'train')
test_folder = os.path.join(base_folder, 'test')

# Get a list of subdirectories (classes) in the train and test folders
train_classes = set([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
test_classes = set([d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))])

# Find classes in train but not in test
classes_to_create = train_classes - test_classes

# Create folders in test for missing classes
for class_name in classes_to_create:
    test_class_path = os.path.join(test_folder, class_name)
    os.makedirs(test_class_path, exist_ok=True)

print("Folders created in test for classes not present in test.")
