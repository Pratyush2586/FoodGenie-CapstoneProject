import pandas as pd
import numpy as np
import re
import os
import shutil

import warnings
warnings.filterwarnings("ignore")


data_val = pd.read_csv("val_data.csv")


food_title_list = ['The Best Rolled Sugar Cookies',
 'Marshmallow Fondant',
 'Apple Pie by Grandma Ople',
 'To Die For Blueberry Muffins',
 "Chantal's New York Cheesecake",
 'Cake Balls',
 'Clone of a Cinnabon',
 'Banana Banana Bread',
 "World's Best Lasagna",
 'Carrot Cake III',
 'Award Winning Soft Chocolate Chip Cookies',
 'Bomb Ass Potatoes',
 'Amish White Bread',
 'Bacon Wrapped Chicken Breast',
 'White Chocolate Raspberry Cheesecake',
 'Parmesan Chicken Squares',
 'Good Old Fashioned Pancakes',
 'Easy Sugar Cookies',
 'Best Brownies',
 'Cajun Chicken Pasta',
 'Delicious Ham and Potato Soup',
 'Chicken Parmesan',
 'Peanut Butter Cup Cookies',
 'Downeast Maine Pumpkin Bread',
 'Black Magic Cake',
 "Rick's Special Buttercream Frosting",
 'Cream Cheese Frosting II',
 'Special Buttercream Frosting',
 "Grandma's Lemon Meringue Pie",
 'Simple White Cake',
 'Big Soft Ginger Cookies',
 'Easy OREO Truffles',
 'Soft Oatmeal Cookies',
 'Tiramisu Layer Cake',
 'Baked MAC and Cheese',
 'Ninety Minute Cinnamon Rolls',
 'Bread Pudding II',
 "Spooky Witches' Fingers",
 'Easy Chicken with Broccoli',
 "Mom's Zucchini Bread",
 'Chicken Cordon Bleu II',
 'Sweet, Sticky and Spicy Chicken',
 'Red Velvet Cupcakes',
 'Broiled Tilapia Parmesan',
 'Alfredo Sauce']

def create_class_folder(df,folder_path_img,original_path):
    
    for cleaned_title in df['food_title']:
        folder_path = os.path.join(folder_path_img, cleaned_title)
        os.makedirs(folder_path, exist_ok=True)

    # Copy images into the corresponding folders
    for index, row in df.iterrows():
        cleaned_title = row['food_title']
        folder_path = os.path.join(folder_path_img, cleaned_title)

        # Split the images_id and copy each image to the folder
        for image_filename in row['images_id'].split(','):
            image_filename = image_filename.strip()
            src_path = os.path.join(original_path, image_filename)
            dest_path = os.path.join(folder_path, image_filename)
            shutil.copy(src_path, dest_path)

    print("Images copied to folders based on 'food_title'")





# Path to the 'test' folder
folder_path_img = 'train_dataset'
original_path = 'E:/Masters/2024/498/Recipe1M/Images/train_images'


create_class_folder(data_val,folder_path_img,original_path)