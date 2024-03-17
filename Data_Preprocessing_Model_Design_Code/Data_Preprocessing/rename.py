import os

def update_image_names(directory_path, word_to_add):
    # Get the list of files in the directory
    files = os.listdir(directory_path)

    # Filter only image files (you can customize the extension if needed)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Iterate through each image file and rename it
    for image_file in image_files:
        # Create the new filename by adding the word at the start
        new_name = word_to_add + image_file

        # Construct the full paths
        old_path = os.path.join(directory_path, image_file)
        new_path = os.path.join(directory_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {old_path} -> {new_path}')

# Example usage:
directory_path = 'E:/Masters/2024/498/Feature2/Feature_Ingridents/test/Pepper/capsicum'
word_to_add = 'capsicum_'  # Change this to the word you want to add
update_image_names(directory_path, word_to_add)
