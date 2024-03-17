import os

base_folder = r'E:\Masters\2024\498\Feature2\Feature_Ingridents'
train_folder = os.path.join(base_folder, 'train')
test_folder = os.path.join(base_folder, 'test')

# Function to count images in a folder for each class
def count_images(folder):
    classes_count = {}
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            images_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            classes_count[class_name] = images_count
    return classes_count

# Count images in train folder
train_classes_count = count_images(train_folder)

# Count images in test folder
test_classes_count = count_images(test_folder)

# Print the counts for each class
print("Train folder:")
for class_name, count in train_classes_count.items():
    print(f"{class_name}: {count} images")

print("\nTest folder:")
for class_name, count in test_classes_count.items():
    print(f"{class_name}: {count} images")
