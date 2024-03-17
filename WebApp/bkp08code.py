import streamlit as st
import pandas as pd
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image
import re
import os
import pickle
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
#from classify_Indian_dish import classify_Indian_dish
openai.api_key  = os.environ["OPENAI_API_KEY"]

# Initialize the OpenAI client with the API key
client = openai.OpenAI()
# Disable warnings
# Disable warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

import json

# Load nutrition data from JSON file
def load_nutrition_data():
    with open("nutrition.json", "r") as file:
        return json.load(file)

# Get nutritional values for a specific dish class
def get_nutritional_values(nutrition_data, dish_class):
    return nutrition_data.get(dish_class, {})

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def click_button(options):
    st.session_state.choice = options  # Use the variable options directly
    st.query_params['menu_choice'] = options  # Set new query params    
    print("button " + st.session_state.choice + " clicked")
    #st.rerun()


# Initialize session state attributes if they don't already exist
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False
    st.query_params['menu_choice'] = 'None'  # Set new query params

#def display_logo():
    # Create a layout with three columns
#    left_space, logo_column, right_space = st.columns([1, 0.85, 1])  # Adjust the ratio as needed

def display_logo():
    # Create a layout with columns. Use the first column as a spacer and the second one for the logo.
    spacer_column, logo_column = st.columns([0.1, 9.9])  # Adjust the ratio for finer control
    
    with logo_column:
        # Assuming you have an image named 'logo.jpg' in your working directory
        st.image("cap1.jpg", width=100)  # Adjust the path and width as needed

def display_logout_button():
    # Layout with columns for spacer, spacer, and logout button
    _, _, logout_column = st.columns([0.6, 0.1, 0.1])
    #spacer_column, logout_column = st.columns([0.95, 0.05])
    with logout_column:
        if st.button("Log Out"):
            st.session_state['is_logged_in'] = False
            # Optionally reset other session states
            for key in list(st.session_state.keys()):
                if key != 'is_logged_in':
                    del st.session_state[key]
            st.experimental_rerun()

# Function to classify image and display recipe for Indian dishes

# Helper function to format ingredients

# Function to classify image and display recipe for continental/confectionery dishes
#function end for display
def format_ingredients(ingredients):
    # Improved ingredients formatting logic
    ingredients = re.sub(r'^-\s*', '', ingredients)
    ingredients_list = re.split(r',\s*', ingredients)
    formatted_ingredients = '<ul>' + ''.join(f'<li>{ing.strip()}</li>' for ing in ingredients_list if ing) + '</ul>'
    return formatted_ingredients

def format_instructions(instructions):
    """
    Formats the instructions text into an HTML unordered list for better readability.
    """
    # Remove any unwanted leading characters, like '1. '
    instructions = re.sub(r'^1\.\s*', '', instructions)
    # Split instructions into individual steps. Assumes steps are separated by periods followed by a space or end of the string.
    instructions_steps = re.split(r'\.\s+|\.$', instructions)
    # Filter out any empty strings that might be created by the split operation.
    instructions_steps = [step for step in instructions_steps if step]
    # Create an unordered list (HTML) of instruction steps.
    formatted_instructions = '<ul>' + ''.join(f'<li>{step.strip()}.</li>' for step in instructions_steps) + '</ul>'
    return formatted_instructions

# Function to format nutritional values
def format_nutritional_values(nutrition_content):
    formatted_values = ""
    for key, value in nutrition_content.items():
        if key != 'name' and key != 'category':
            formatted_values += f"'{key}': {value}, "
    return formatted_values[:-2]  # Remove the extra comma and space at the end

##display function
def display_contrecipe_details(ingredients, instructions):
    """
    Display the ingredients and instructions for continental recipes with custom styling to evoke the feel of an ancient manuscript.
    """
    # Apply custom styling for a manuscript-like presentation
    st.markdown("""
        <style>
            .infoBox {
                font-family: 'Times New Roman', Times, serif; /* Manuscript-like font */
                color: #5C3A1B; /* Deep brown text for an old manuscript look */
                background-color: #F5EEDC; /* Light papyrus-like background color */
                border: solid 2px #A0522D; /* Border to mimic the edges of a scroll */
                border-radius: 0; /* Scrolls typically do not have rounded corners */
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 0 10px rgba(0,0,0,0.5); /* Soft shadow for depth */
                overflow-wrap: break-word; /* Ensure long words do not overflow */
                text-align: left;
            }
            .infoTitle {
                font-size: 24px; /* Larger title font for headings */
                font-weight: bold;
                color: #8B572A; /* Darker brown for titles to stand out */
                margin-bottom: 15px;
            }
            ul, ol {
                padding-left: 20px;
            }
            li {
                margin-bottom: 10px;
                line-height: 1.6;
            }
            li strong {
                font-weight: bold;
                color: #8B4513; /* Color for subheaders */
            }
            .goToMainMenuButton {
                margin-top: 20px;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

    # Use columns to display Ingredients and Instructions side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div class='infoBox'><div class='infoTitle'>Ingredients</div>{ingredients}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='infoBox'><div class='infoTitle'>Instructions</div>{instructions}</div>", unsafe_allow_html=True)
    
    # Add a button to go back to the main menu
        # Add a button to go back to the main menu
        st.button('Go to Main Menu', on_click=click_button, args=('option3',))

    #st.markdown("<div class='goToMainMenuButton'><a href='/home'><button>Go to Main Menu</button></a></div>", unsafe_allow_html=True)



## new function for format stars
def display_openairecipeonly_details(recipe):
    """
    Display the generated recipe with custom styling to evoke the feel of an ancient manuscript.
    """
    # Apply custom styling for a manuscript-like presentation
    st.markdown("""
        <style>
            .infoBox {
                font-family: 'Times New Roman', Times, serif; /* Manuscript-like font */
                color: #5C3A1B; /* Deep brown text for an old manuscript look */
                background-color: #F5EEDC; /* Light papyrus-like background color */
                border: solid 2px #A0522D; /* Border to mimic the edges of a scroll */
                border-radius: 0; /* Scrolls typically do not have rounded corners */
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 0 10px rgba(0,0,0,0.5); /* Soft shadow for depth */
                overflow-wrap: break-word; /* Ensure long words do not overflow */
                text-align: left;
            }
            .infoTitle {
                font-size: 24px; /* Larger title font for headings */
                font-weight: bold;
                color: #8B572A; /* Darker brown for titles to stand out */
                margin-bottom: 15px;
            }
            li {
                margin-bottom: 10px;
                line-height: 1.6;
            }
            .goToMainMenuButton {
                margin-top: 20px;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

    # Display the generated recipe
    st.markdown(f"<div class='infoBox'><div class='infoTitle'>Recipe</div>{recipe}</div>", unsafe_allow_html=True)

    # Add a button to go back to the main menu
    st.button('Go to Main Menu', on_click=click_button, args=('option3',))

##function end 
def display_recipe_details(ingredients, instructions, video_link, formatted_values=None):
    """
    Display the ingredients, instructions, video link, and formatted nutrition values for recipes.
    """
    # Apply custom styling for a manuscript-like presentation
    st.markdown("""
        <style>
            .infoBox {
                font-family: 'Times New Roman', Times, serif; /* Manuscript-like font */
                color: #5C3A1B; /* Deep brown text for an old manuscript look */
                background-color: #F5EEDC; /* Light papyrus-like background color */
                border: solid 2px #A0522D; /* Border to mimic the edges of a scroll */
                border-radius: 0; /* Scrolls typically do not have rounded corners */
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 0 10px rgba(0,0,0,0.5); /* Soft shadow for depth */
                overflow-wrap: break-word; /* Ensure long words do not overflow */
                text-align: left;
            }
            .infoTitle {
                font-size: 24px; /* Larger title font for headings */
                font-weight: bold;
                color: #8B572A; /* Darker brown for titles to stand out */
                margin-bottom: 15px;
            }
            ul, ol {
                padding-left: 20px;
            }
            li {
                margin-bottom: 10px;
                line-height: 1.6;
            }
            li strong {
                font-weight: bold;
                color: #8B4513; /* Color for subheaders */
            }
            .videoContainer {
                position: relative;
                width: 100%;
                padding-bottom: 56.25%; /* 16:9 aspect ratio */
            }
            .videoFrame {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
            /* Add padding around each frame */
            .framePadding {
                padding: 20px;
            }
        </style>
        """, unsafe_allow_html=True)

    # Use columns to display Ingredients, Instructions, Video, and Nutrition side by side
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        st.markdown(f"<div class='infoBox framePadding'><div class='infoTitle'>Ingredients</div>{ingredients}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='infoBox framePadding'><div class='infoTitle'>Instructions</div>{instructions}</div>", unsafe_allow_html=True)

    with col3:
        if formatted_values:
            st.markdown(f"<div class='infoBox'><div class='infoTitle'>Nutrition</div>{formatted_values}</div>", unsafe_allow_html=True)
        if video_link:
            st.markdown(f"<div class='infoBox'><div class='infoTitle'>Video</div><div class='videoContainer'><iframe class='videoFrame' src='{video_link}' frameborder='0' allowfullscreen></iframe></div></div>", unsafe_allow_html=True)
        else:
            st.warning("No video link available for this recipe.")

    # Add a button to go back to the main menu
    st.button('Go to Main Menu', on_click=click_button, args=('option3',))


def classify_continental_dish(uploaded_file):
    st.session_state.show_details = False
    # Main function for classifying and displaying recipe details
    #st.write("Processing the uploaded Continental/Confectionery dish image...")
    try:
        # Configuration settings for Azure Custom Vision
        prediction_endpoint = 'https://foodgeniecv-prediction.cognitiveservices.azure.com/'
        prediction_key = 'a766c4271ccd40fbbda4ca11ca1406a2'  # Replace with your actual key
        project_id = 'bc56c2ed-4d0f-4eb1-ba5c-e6ce106bfc8c'  # Replace with your actual project ID
        model_name = 'Foodgenie_Recipe1M'  # Replace with your actual model name for continental/confectionery dishes

        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)
        # Create an empty element to be replaced with the message when the image is being processed
        message_placeholder = st.empty()
        message_placeholder.write("FoodGenie is Processing Your uploaded  dish image...")
        # Perform prediction

        results = prediction_client.classify_image(project_id, model_name, uploaded_file.read())

        max_prob, max_class = max(((prediction.probability, prediction.tag_name) for prediction in results.predictions), default=(0, None))

        if max_class and max_prob > 0.8:
            message_placeholder.empty()  # Remove the processing message
            st.markdown(f"### Voilà! FoodGenie has magically identified your dish as <span style='color: blue;'>{max_class}</span>. Ready to dive into delicious recipes? Click below!", unsafe_allow_html=True)

            if 'show_details' not in st.session_state or not st.session_state.show_details:
                if st.button('Show Ingredients and Instructions'):
                    st.session_state.show_details = True

            if 'show_details' in st.session_state and st.session_state.show_details:
                df = pd.read_csv("recipe_with_nutrients_cleaned.csv")  # Make sure this path is correct
                idx = df.index[df['food_title'] == max_class].tolist()

                if idx:
                    ingredients = format_ingredients(df.loc[idx[0], 'ingredients_unique'])
                    ingredients = ingredients.replace('-', '', 1)
                    instructions = format_instructions(df.loc[idx[0], 'instructions_unique'])
                    instructions = instructions.replace(re.search(r'\d+\.', instructions).group(), '', 1)
                    display_contrecipe_details(ingredients, instructions)
                else:
                    st.error("No information found for the identified dish.")
        else:
            st.error("Oops! The Food Genie is on a learning spree. Please try a different food image.")
    except Exception as ex:
        st.error(f"Error: {ex}")

def classify_Indian_dish(uploaded_file):
    st.session_state.show_details = False
    #st.write("Processing the uploaded Indian dish image...")
    try:
        # Configuration Settings (Replace with your actual settings)
        # Set Configuration Settings
        prediction_endpoint = 'https://foodgenieindian-prediction.cognitiveservices.azure.com/'
        prediction_key = '4cf728387209453c8b903d93a7416597'  # Replace with your actual key
        project_id = 'f89acc57-25d2-411f-9558-be455d2469e2'  # Replace with your actual project ID
        model_name = 'FoodGenieIndian'  # Replace with your actual model name for Indian dishes

        # Authenticate and create prediction client
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)
        # Create an empty element to be replaced with the message when the image is being processed
        message_placeholder = st.empty()
        message_placeholder.write("FoodGenie is Processing Your uploaded  dish image...")
        # Perform prediction
        results = prediction_client.classify_image(project_id, model_name, uploaded_file.read())

        # Process prediction results
        max_prob, max_class = max(((prediction.probability, prediction.tag_name) for prediction in results.predictions), default=(0, None))

        if max_class and max_prob > 0.8:
            message_placeholder.empty()  # Remove the processing message
            st.markdown(f"### Voilà! FoodGenie has magically identified your dish as <span style='color: blue;'>{max_class}</span>. Ready to dive into delicious recipes? Click below!", unsafe_allow_html=True)

            if 'show_details' not in st.session_state or not st.session_state.show_details:
                if st.button('Show Ingredients and Instructions'):
                    st.session_state.show_details = True

            if 'show_details' in st.session_state and st.session_state.show_details:
                df = pd.read_csv("recipe_with_indian_cleaned.csv")  # Make sure this path is correct
                idx = df.index[df['food_title'] == max_class].tolist()

                if idx:
                    ingredients = format_ingredients(df.loc[idx[0], 'ingredients_unique'])
                    ingredients = ingredients.replace('-', '', 1)
                    instructions = format_instructions(df.loc[idx[0], 'instructions_unique'])
                    instructions = instructions.replace(re.search(r'\d+\.', instructions).group(), '', 1)
                    video_link = df.loc[idx[0], 'Video_link']  # Extracting video link from CSV
                    nutrition_data = load_nutrition_data()
                    nutrition_content = get_nutritional_values(nutrition_data, max_class)
                    formatted_values = format_nutritional_values(nutrition_content)
                    print(formatted_values)
                    display_recipe_details(ingredients, instructions, video_link,formatted_values)
                else:
                    st.error("No information found for the identified dish.")
        else:
            st.error("Oops! The Food Genie is on a learning spree. Please try a different food image.")
    except Exception as ex:
        st.error(f"Error: {ex}")
##main Ingredients Function
def classify_Ingredients_dish(uploaded_file):
    st.session_state.show_details = False
    #st.write("Processing the uploaded Indian dish image...")
    try:
        # Configuration Settings (Replace with your actual settings)
        # Configuration settings for Azure Custom Vision
        prediction_endpoint = 'https://foodgeniecv-prediction.cognitiveservices.azure.com/'
        prediction_key = 'a766c4271ccd40fbbda4ca11ca1406a2'  # Replace with your actual key
        project_id = '3deac33d-ab06-43cb-bbd9-65e6219d8f19'  # Replace with your actual project ID
        model_name = 'IngredientsTrial'  # Replace with your actual object detection model name

        # Authenticate and create prediction client
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

        # Perform object detection
        #results = prediction_client.detect_image(project_id, model_name, uploaded_file.read())

        # Filter predictions with probability higher than 45%
        #high_prob_predictions = [prediction for prediction in results.predictions if prediction.probability > 0.30]

        # Extract the tag names of high probability predictions
        #max_classes = "|".join(prediction.tag_name for prediction in high_prob_predictions)
                # Perform object detection
        # Create an empty element to be replaced with the message when the image is being processed
        message_placeholder = st.empty()
        message_placeholder.write("FoodGenie is Processing Your uploaded  Ingredients image...")
        # Perform prediction                
        results = prediction_client.detect_image(project_id, model_name, uploaded_file.read())

        # Filter predictions with probability higher than 45%
        high_prob_predictions = [prediction for prediction in results.predictions if prediction.probability > 0.30]

        # Extract unique tag names of high probability predictions
        unique_classes = set(prediction.tag_name for prediction in high_prob_predictions)

        # Join unique classes separated by pipe '|'
        max_classes = "|".join(unique_classes)

        if max_classes:
            message_placeholder.empty()  # Remove the processing message
            st.markdown(f"### Voilà! FoodGenie has magically identified your Ingredients as <span style='color: blue;'>{max_classes}</span>. Ready to dive into delicious recipes? Click below!", unsafe_allow_html=True)

            if 'show_details' not in st.session_state or not st.session_state.show_details:
                if st.button('Show Ingredients and Instructions'):
                    st.session_state.show_details = True

            if 'show_details' in st.session_state and st.session_state.show_details:
                #df = pd.read_csv("recipe_with_ingredients.csv")  # Make sure this path is correct
                df = pd.read_csv("recipe_with_ingredients.csv", encoding='ISO-8859-1')
                matching_dishes = df[df['food_title'] == max_classes]
                print(max_classes)
                #for max_class in unique_classes:
                #    idx = df.index[df['food_title'] == max_classes].tolist()
                #    print(idx)
                if not matching_dishes.empty:
                  idx = matching_dishes.index[0]
                  ingredients = format_ingredients(df.loc[idx, 'ingredients_unique'])
                  instructions = format_instructions(df.loc[idx, 'instructions_unique'])
                  display_recipe_details(ingredients, instructions)
                else:
                    st.error("No information found for the identified dish.")
        else:
            st.error("Oops! The Food Genie is on a learning spree. Please try a different food image.")
    except Exception as ex:
        st.error(f"Error: {ex}")


## Ingredients end
def classify_Ingredients_dish_openai(uploaded_file):
    st.session_state.show_details = False
    # Initialize the OpenAI client with the API key
    #client = openai.OpenAI()
    #st.write("Processing the uploaded Indian dish image...")
    try:
        # Configuration settings for Azure Custom Vision
        prediction_endpoint = 'https://foodgeniecv-prediction.cognitiveservices.azure.com/'
        prediction_key = 'a766c4271ccd40fbbda4ca11ca1406a2'  # Replace with your actual key
        project_id = '3deac33d-ab06-43cb-bbd9-65e6219d8f19'  # Replace with your actual project ID
        model_name = 'Iteration4'  # Replace with your actual object detection model name

        # Authenticate and create prediction client
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

        # Perform object detection
        #results = prediction_client.detect_image(project_id, model_name, uploaded_file.read())
        # Filter predictions with probability higher than 30%
        #high_prob_predictions = [prediction for prediction in results.predictions if prediction.probability > 0.30]
        # Extract unique tag names of high probability predictions
        #unique_classes = set(prediction.tag_name for prediction in high_prob_predictions)
        # Join unique classes separated by pipe '|'
        #max_classes = "|".join(unique_classes)
        # Perform object detection
        message_placeholder = st.empty()
        message_placeholder.write("FoodGenie is Processing Your uploaded  Ingredients image...")
        # Perform prediction
        results = prediction_client.detect_image(project_id, model_name, uploaded_file.read())

        # Extract all unique tag names
        all_classes = set(prediction.tag_name for prediction in results.predictions)

        # Sort the classes based on their occurrence in the predictions
        sorted_classes = sorted(all_classes, key=lambda x: sum(1 for prediction in results.predictions if prediction.tag_name == x), reverse=True)

        # Select the top three unique ingredient classes
        top_three_classes = sorted_classes[:3]

        # Join top three classes separated by pipe '|'
        max_classes = "|".join(top_three_classes)        

        if max_classes:
            message_placeholder.empty()  # Remove the processing message
            st.markdown(f"### Voilà! FoodGenie has magically identified your Ingredients as <span style='color: blue;'>{max_classes}</span>. Ready to dive into delicious recipes? Click below!", unsafe_allow_html=True)

            if 'show_details' not in st.session_state or not st.session_state.show_details:
                if st.button('Show Ingredients and Instructions'):
                    st.session_state.show_details = True

            if 'show_details' in st.session_state and st.session_state.show_details:
                # Use OpenAI ChatGPT to generate a recipe based on the identified ingredients
                #openai.api_key = "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"  # Replace with your OpenAI API key
                #prompt = f"Could you please provide me one recipe that I can cook using the ingredients: {max_classes}?"
                prompt = f"Could you please provide me two quick recipe that I can cook using the ingredients: {max_classes}?.Please start the answer with FoodGenie brings you the below recipe with your selected Ingredients: and also each recipe should start with Recipe1:Recipe_name followed by Ingredients and Instructions to cook ,Recipe2:Recipe_name followed by Ingredients and Instructions in proper format."
                recipe = get_completion(prompt)
                #recipe = response.choices[0].text.strip()

                # Display the generated recipe
                st.markdown(f"#### Recipe Generated:")
                #st.write(recipe)
                #st.markdown(f"<div style='text-align: center;'>{recipe}</div>", unsafe_allow_html=True)
                # Extract ingredients and instructions from the recipe
                # For example, you can use regex or string manipulation to extract them
                # For demonstration, assuming the ingredients are before a certain keyword and instructions are after
                #ingredients = recipe.split("Ingredients:")[1].split("Instructions:")[0].strip()
                #instructions = recipe.split("Instructions:")[1].strip()

    # Display the ingredients, instructions, and generated recipe using custom function
                #display_openairecipe_details(ingredients=ingredients, instructions=instructions, recipe=recipe)
                display_openairecipeonly_details(recipe)

        else:
            st.error("Oops! The Food Genie is on a learning spree. Please try a different food image.")

    except Exception as ex:
        st.error(f"Error: {ex}")
# Helper function to format ingredients
def format_ingredients(ingredients):
    formatted_ingredients = ""
    for ingredient in ingredients.split('\n'):
        formatted_ingredients += f"- {ingredient.strip()}\n"
    return formatted_ingredients

# Helper function to format instructions
def format_instructions(instructions):
    formatted_instructions = ""
    for i, instruction in enumerate(instructions.split('\n'), start=1):
        formatted_instructions += f"{i}. {instruction.strip()}\n"
    return formatted_instructions

# Page definitions

# Make sure to define the display_logo and verify_login functions accordingly
import base64

# Function to get base64 encoded string for an image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Function to set the background image
def set_background_image(image_path):
    background_image_base64 = get_base64_encoded_image(image_path)
    # Set page background using the base64 encoded image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
            background-attachment: fixed; /* Prevent background from scrolling */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Your existing login page function        

def login_page():
    # Encode your image to base64
    display_logo()  # Display the logo
    background_image_base64 = get_base64_encoded_image("callmeGenie.jpg")
    
    # Set page background using the base64 encoded image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
        }}
        .stTextInput>div>div>input {{
            color: #000000; /* Bold and black text color for input fields */
            font-weight: bold; /* Ensure text is bold */
            background-color: rgba(255, 255, 255, 0.8); /* Light background for input fields */
            background-attachment: fixed; /* Prevent background from scrolling */
            overflow: hidden; /* Hide overflow to prevent scroll */
            border-radius: 15px; /* Slightly rounded corners */
            border: 1px solid #ced4da; /* Light border color */
            padding: 8px; /* Padding inside input fields */
        }}
        .stTextInput>label {{
            color: #333; /* Darker color for better visibility */
            font-weight: bold; /* Make sure the label text is bold */
            font-size: 16px; /* Slightly larger font size for better readability */
        }}
        .stButton>button {{
            display: block;
            width: 40%; /* Adjust button width */
            margin: 20px auto; /* Center button horizontally */
            border-radius: 15px; /* Rounded corners for the button */
            border: 1px solid transparent; /* No border for the button */
            padding: 10px 24px; /* Padding inside the button */
            background-color: #4CAF50; /* Button background color */
            color: white; /* Button text color */
        }}
        h1 {{
            color: #000000; /* Orange color for the header */
            text-align: center;
        }}
        /* Ensure the form container fits within the viewport without scrolling */
        .stContainer {{
            max-height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("####")  # Add more of these as needed to increase vertical space
    st.markdown("####")  # Add more of these as needed to increase vertical space
    st.markdown("####")  # Add more of these as needed to increase vertical space,
    st.markdown("####")  # Add more of these as needed to increase vertical space
    st.markdown("<h1>Welcome to FoodGenie</h1>", unsafe_allow_html=True)
    # Add vertical space before the form
    #st.markdown("####")  # Add more of these as needed to increase vertical space
    #st.markdown("####")  # Add more of these as needed to increase vertical space
    # Adjusting the form layout
    col1, form, col3 = st.columns([1.2, 1.6, 1.2])
    # Pre-fill username and password
    #default_username = "pratyush"
    #default_password = "pratyush123"

    with form:
        #username = st.text_input("Username",value=default_username, key="username_login")
        #password = st.text_input("Password", type="password",value=default_password, key="password_login")
        username = st.text_input("Username", key="username_login")
        password = st.text_input("Password", type="password", key="password_login")
        if st.button("Login"):
            if verify_login(username, password):
                st.session_state.is_logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

# Replace this with your actual main page content function
# Function to verify login credentials
# Function to get base64 encoded image
def get_base64_encoded_image1(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Set background image
newbackground_image_base64 = get_base64_encoded_image1("callmeGenie.jpg")

def classify_Indian_dish_page():
    # Set background image and styles
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
        }}
        .stTextInput>div>div>input[type="file"] {{
            color: #000000; /* Bold and black text color for input fields */
            font-weight: bold; /* Ensure text is bold */
            background-color: rgba(255, 255, 255, 0.8); /* Light background for input fields */
            background-attachment: fixed; /* Prevent background from scrolling */
            overflow: hidden; /* Hide overflow to prevent scroll */
            border-radius: 15px; /* Slightly rounded corners */
            border: 1px solid #ced4da; /* Light border color */
            padding: 8px; /* Padding inside input fields */
            width: 50%; /* Set width to 50% of the container */
            margin: 0 auto; /* Center horizontally */
            display: block; /* Make it block-level element */
            text-align: center; /* Center text */
        }}
        .stTextInput>label {{
            color: #333; /* Darker color for better visibility */
            font-weight: bold; /* Make sure the label text is bold */
            font-size: 16px; /* Slightly larger font size for better readability */
        }}
        .stButton>button {{
            display: block;
            width: 40%; /* Adjust button width */
            margin: 20px auto; /* Center button horizontally */
            border-radius: 15px; /* Rounded corners for the button */
            border: 1px solid transparent; /* No border for the button */
            padding: 10px 24px; /* Padding inside the button */
            background-color: #4CAF50; /* Button background color */
            color: white; /* Button text color */
        }}
        h1 {{
            color: #000000; /* Orange color for the header */
            text-align: center;
        }}
        /* Ensure the form container fits within the viewport without scrolling */
        .stContainer {{
            max-height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h3>Upload your image for Indian Dishes:</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop file here or Browse files", type=["jpg", "png", "jpeg"], key="indian_dish_uploader", help="Supported formats: JPG, PNG, JPEG", label_visibility="collapsed", accept_multiple_files=False)
    if uploaded_file is not None:
        print("eneterd here")
        st.image(uploaded_file, caption="Uploaded Image", width=390)  # Adjust width as needed                     
        classify_Indian_dish(uploaded_file)
        #classify_Indian_dish(uploaded_file)
# function end---

def classify_continental_dish_page():
        # Set background image and styles
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
        }}
        .stTextInput>div>div>input[type="file"] {{
            color: #000000; /* Bold and black text color for input fields */
            font-weight: bold; /* Ensure text is bold */
            background-color: rgba(255, 255, 255, 0.8); /* Light background for input fields */
            background-attachment: fixed; /* Prevent background from scrolling */
            overflow: hidden; /* Hide overflow to prevent scroll */
            border-radius: 15px; /* Slightly rounded corners */
            border: 1px solid #ced4da; /* Light border color */
            padding: 8px; /* Padding inside input fields */
            width: 50%; /* Set width to 50% of the container */
            margin: 0 auto; /* Center horizontally */
            display: block; /* Make it block-level element */
            text-align: center; /* Center text */
        }}
        .stTextInput>label {{
            color: #333; /* Darker color for better visibility */
            font-weight: bold; /* Make sure the label text is bold */
            font-size: 16px; /* Slightly larger font size for better readability */
        }}
        .stButton>button {{
            display: block;
            width: 40%; /* Adjust button width */
            margin: 20px auto; /* Center button horizontally */
            border-radius: 15px; /* Rounded corners for the button */
            border: 1px solid transparent; /* No border for the button */
            padding: 10px 24px; /* Padding inside the button */
            background-color: #4CAF50; /* Button background color */
            color: white; /* Button text color */
        }}
        h1 {{
            color: #000000; /* Orange color for the header */
            text-align: center;
        }}
        /* Ensure the form container fits within the viewport without scrolling */
        .stContainer {{
            max-height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h3>Upload your image for Continental/Confectionery Dishes:</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop file here or Browse files", type=["jpg", "png", "jpeg"], key="continental_dish_uploader", help="Supported formats: JPG, PNG, JPEG", label_visibility="collapsed", accept_multiple_files=False)
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=390)
        # Assuming classify_continental_dish function is defined to handle the uploaded file
        classify_continental_dish(uploaded_file)

#Define function for Ingredients Call
def classify_Ingredients_dish_page():
        # Set background image and styles
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
        }}
        .stTextInput>div>div>input[type="file"] {{
            color: #000000; /* Bold and black text color for input fields */
            font-weight: bold; /* Ensure text is bold */
            background-color: rgba(255, 255, 255, 0.8); /* Light background for input fields */
            background-attachment: fixed; /* Prevent background from scrolling */
            overflow: hidden; /* Hide overflow to prevent scroll */
            border-radius: 15px; /* Slightly rounded corners */
            border: 1px solid #ced4da; /* Light border color */
            padding: 8px; /* Padding inside input fields */
            width: 50%; /* Set width to 50% of the container */
            margin: 0 auto; /* Center horizontally */
            display: block; /* Make it block-level element */
            text-align: center; /* Center text */
        }}
        .stTextInput>label {{
            color: #333; /* Darker color for better visibility */
            font-weight: bold; /* Make sure the label text is bold */
            font-size: 16px; /* Slightly larger font size for better readability */
        }}
        .stButton>button {{
            display: block;
            width: 40%; /* Adjust button width */
            margin: 20px auto; /* Center button horizontally */
            border-radius: 15px; /* Rounded corners for the button */
            border: 1px solid transparent; /* No border for the button */
            padding: 10px 24px; /* Padding inside the button */
            background-color: #4CAF50; /* Button background color */
            color: white; /* Button text color */
        }}
        h1 {{
            color: #000000; /* Orange color for the header */
            text-align: center;
        }}
        /* Ensure the form container fits within the viewport without scrolling */
        .stContainer {{
            max-height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h3>Upload your image for the Ingredients You have:</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop file here or Browse files", type=["jpg", "png", "jpeg"], key="continental_dish_uploader", help="Supported formats: JPG, PNG, JPEG", label_visibility="collapsed", accept_multiple_files=False)
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=390)
        # Assuming classify_continental_dish function is defined to handle the uploaded file
        #classify_Ingredients_dish(uploaded_file)
        classify_Ingredients_dish_openai(uploaded_file)
#####                

# Function to resize image
def resize_image(image_path, width, height):
    image = Image.open(image_path)
    resized_image = image.resize((width, height))
    return resized_image

# Function to navigate to the main menu page
def main_menu_page():
    display_logo()  # Display the logo at the top
    print("entered main_menu_page ")
    # Display the logout button
    display_logout_button()
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #0E1117; font-size: 34px; font-weight: bold; margin-bottom: 20px;">
                Welcome to FoodGenie
            </h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        /* Ensure buttons are centered and match the width of images */
        .stButton>button {{
            border-radius: 20px;
            border: 1px solid #4CAF50;
            color: white;
            background-color: #4CAF50;
            cursor: pointer;
            padding: 10px;
            font-size: 18px;
            line-height: 20px;
            display: inline-block; /* Center button */
            margin: 0 auto; /* Center button horizontally */
            width: 300px; /* Match the button width to the image width */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <div style="text-align: center;">
            <h2 style="color: #0E1117; font-size: 28px; font-weight: bold; margin-bottom: 20px;">
                Select your Options
            </h2>
        </div>
    """, unsafe_allow_html=True)


    # Adjust columns for alignment
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        col1, mid_col, col2 = st.columns([3, 2, 3])

        with col1:
            resized_image1 = resize_image("dish2.jpg", 270, 270)
            st.image(resized_image1, use_column_width=True)
            st.button('Reveal Recipes from My Dish', on_click=click_button, args=('option1',))


        with col2:
            resized_image2 = resize_image("ingredients1.jpg", 270, 270)
            st.image(resized_image2, use_column_width=True)            
            st.button('Craft a Dish from Your Ingredients!', on_click=click_button, args=('option2',))

    # Place the "Return to Home" button after the columns for options
    #st.markdown("""
    #    <div style="text-align: center; margin-top: 20px;">
    #        <a href="/home" style="text-decoration: none;">
    #            <button style="border-radius: 20px; border: 1px solid #4CAF50; color: white; background-color: #4CAF50; cursor: pointer; padding: 10px 20px; font-size: 18px; line-height: 20px;">
    #                Return to Home
    #            </button>
    #        </a>
    #    </div>
    #""", unsafe_allow_html=True)




# Global variable to hold the base64 encoded background image
background_image_base64 = get_base64_encoded_image("background.jpg")

# Main page function
def main_page():
    display_logo()  # Display the logo at the top
    print("entered main page")
    # Apply a background image with CSS for a semi-transparent overlay
    background_image_base64 = get_base64_encoded_image("background.jpg")

    # Display the background image and title
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        /* Button styling */
        .stButton>button {{
            border: 2px solid #4CAF50;
            border-radius: 20px;
            color: white;
            background-color: #4CAF50;
            cursor: pointer;
            padding: 10px 24px;
            font-size: 18px;
            display: block;
            margin: 10px auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #0E1117; font-size: 34px; font-weight: bold; margin-bottom: 20px;">
                Upload Image to Unleash FoodGenie
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Adjust columns for alignment
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        col1, mid_col, col2 = st.columns([3, 2, 3])

        with col1:
            resized_image1 = resize_image("CallIndian_Genie.jpg", 290, 290)
            st.image(resized_image1, use_column_width=True)
            #st.image("CallIndian_Genie.jpg", width=350)
            st.button('Explore Indian Dishes', on_click=click_button, args=('indian',))
            #st.button("Explore Indian Dishes"):

        with col2:
            resized_image2 = resize_image("callContinentalGenie.jpg", 290, 290)
            st.image(resized_image2, use_column_width=True)
            #st.image("callContinentalGenie.jpg", width=350)
            st.button('Explore Continental Dishes', on_click=click_button, args=('continental',))



    #st.rerun()
# Function to verify login credentials
def verify_login(username, password):
    # Placeholder for actual authentication logic
    # Replace with your authentication mechanism
    return username == "pratyush" and password == "pratyush123"

## logout function
def logout_page():
    # Display logout message
    st.markdown("""
        <div style="text-align: center;">
            <h2>You have been successfully logged out</h2>
        </div>
    """, unsafe_allow_html=True)
    # Optionally, you can add a button to log in again
    if st.button('Log In Again'):
        st.session_state['is_logged_in'] = False
        st.experimental_rerun()    

# Main function to organize the app flow
def main():
    st.set_page_config(page_title="FoodGenie", layout="wide")

    if 'is_logged_in' not in st.session_state:
        st.session_state['is_logged_in'] = False
        st.query_params['menu_choice'] = 'None'  # Set new query params    

    if st.session_state['is_logged_in']:
        menu_choice = st.query_params.get("menu_choice", None)  # Use st.query_params with default fallback
        print(menu_choice)
        # Logout option
        if menu_choice == "logout":
            logout_page()
            return  # Ensure no further processing after logout

        # Conditional navigation based on the menu choice
        if menu_choice == "option1":
            main_page()
        elif menu_choice == "option2":
            classify_Ingredients_dish_page()
        elif menu_choice == "indian":
            classify_Indian_dish_page()
        elif menu_choice == "continental":
            classify_continental_dish_page()
        else:
            main_menu_page()
    else:
        login_page()

if __name__ == "__main__":
    main()