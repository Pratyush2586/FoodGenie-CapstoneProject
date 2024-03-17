import os
import shutil

def move_images_to_main_folder(main_folder_path):
    # Ensure the main folder exists
    if not os.path.exists(main_folder_path):
        print(f"The main folder '{main_folder_path}' does not exist.")
        return
    
    # Iterate through all subfolders
    for root, dirs, files in os.walk(main_folder_path):
        for file in files:
            # Check if the file is an image (you can customize the list of image extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Construct the source and destination paths
                source_path = os.path.join(root, file)
                destination_path = os.path.join(main_folder_path, file)
                
                # Move the image file
                shutil.move(source_path, destination_path)
                
                print(f"Moved '{file}' from '{source_path}' to '{destination_path}'.")

# Example usage: specify the path to the main folder
main_folder_path = 'E:/Masters/2024/498/Feature2/Feature_Ingridents/test/Pepper'
move_images_to_main_folder(main_folder_path)
