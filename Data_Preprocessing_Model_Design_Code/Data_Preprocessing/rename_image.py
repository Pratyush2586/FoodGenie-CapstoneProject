from PIL import Image
import os

def convert_and_rename_images(folder_path):
    # Ensure the folder path ends with a '/'
    folder_path = folder_path.rstrip('/') + '/'

    # Create a new folder to store the converted images
    output_folder = folder_path + 'converted_images/'
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the folder
    files = os.listdir(folder_path)

    # Initialize image counter
    image_counter = 1

    # Loop through each file in the folder
    for file_name in files:
        # Check if the file is an image
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Open the image file
            image_path = folder_path + file_name
            img = Image.open(image_path)

            # Convert the image to JPEG format
            new_file_name = f"Coconut_image_{image_counter}.jpg"
            output_path = output_folder + new_file_name
            img.convert("RGB").save(output_path, "JPEG")

            # Increment image counter
            image_counter += 1

    print("Conversion and renaming completed.")

# Example usage
folder_path = 'E:/Masters/2024/498/Feature2/Ingridents_Scrape/Coconut'
convert_and_rename_images(folder_path)
