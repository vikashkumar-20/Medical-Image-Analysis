import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the pre-trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Path to the image you want to predict
image_path = "C:\Users\Vikash\Downloads\Medical Image Analysis Using Generative AI\Brain Tumor Detection - 2.0\pred\pred5.jpg"

# Ensure the image path is correct
if not os.path.exists(image_path):
    print(f"Error: The image path '{image_path}' does not exist.")
    exit()

# Read the image using OpenCV
image = cv2.imread(image_path)

# Ensure the image was read correctly
if image is None:
    print(f"Error: The image at '{image_path}' could not be read.")
    exit()

# Convert the image to RGB format using PIL
img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Resize the image to the input size expected by the model
img = img.resize((64, 64))
img = np.array(img)

# Expand dimensions to match the input shape of the model
input_img = np.expand_dims(img, axis=0)

# Normalize the input image
input_img = input_img / 255.0

# Predict the class
result = model.predict(input_img)

# Get the predicted class
predicted_classes = result.argmax(axis=-1)

# Map the predicted class to the corresponding label
class_labels = {0: "No Brain Tumor", 1: "Yes Brain Tumor"}
predicted_label = class_labels[predicted_classes[0]]

print("Predicted Class:", predicted_classes[0])
print("Predicted Label:", predicted_label)
