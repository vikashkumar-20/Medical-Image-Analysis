import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB as p
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

app = Flask(__name__)

# Set the correct path to the model file
model_path = 'C:/Users/Vikash/Downloads/Medical Image Analysis Using Generative AI/Brain Tumor Detection - 2.0/BrainTumor10EpochsCategorical.h5'

# Load the pre-trained Keras model with error handling
try:
    model = load_model(model_path)
    print('Model loaded. Check http://127.0.0.1:5000/')
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please ensure the path is correct.")
    exit(1)  # Exit the application if the model is not found

# Load accuracy values if files exist
training_accuracy = None
validation_accuracy = None

if os.path.exists('training_accuracy.npy'):
    training_accuracy = np.load('training_accuracy.npy')[-1] * 100  # Convert to percentage
else:
    print("training_accuracy.npy not found")

if os.path.exists('validation_accuracy.npy'):
    validation_accuracy = np.load('validation_accuracy.npy')[-1] * 100  # Convert to percentage
else:
    print("validation_accuracy.npy not found")

def get_class_name(class_no):
    return "Yes Brain Tumor" if class_no == 1 else "No Brain Tumor"

def get_result(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0) / 255.0  # Normalize input image
    result = model.predict(input_img)
    return result.argmax(axis=-1)[0]

# Define machine learning algorithms
def KNN(x_training, y_training, x_testing, y_testing):
    print("\nKNN")
    clf = KNeighborsClassifier(n_neighbors=9)
    clf.fit(x_training, y_training)
    cvs = cross_val_score(clf, x_testing, y_testing, cv=5)
    result = clf.predict(x_testing)
    cmknn = confusion_matrix(y_testing, result)
    accuracy = (cmknn[0][0] + cmknn[1][1]) / len(y_testing) * 100
    return accuracy

def NaiveB(x_training, y_training, x_testing, y_testing):
    print("\nNaive Bayes")
    gnb = p()
    gnb.fit(x_training, y_training)
    y_pre = gnb.predict(x_testing)
    cm = confusion_matrix(y_testing, y_pre)
    accuracy = (cm[0][0] + cm[1][1]) / len(y_testing) * 100
    return accuracy

def DecisionTree(x_training, y_training, x_testing, y_testing):
    print("\nDecision Tree")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_training, y_training)
    y_predict = clf.predict(x_testing)
    CM = confusion_matrix(y_testing, y_predict)
    accuracy = sum(CM.diagonal()) / len(y_testing) * 100
    return accuracy



@app.route('/')
def index():
    # Load the dataset
    d = pd.read_csv("C:/Users/Vikash/Downloads/Medical Image Analysis Using Generative AI/Brain Tumor Detection - 2.0/Brain Tumor.csv")
    
    # Print columns to debug the KeyError
    print(d.columns)
    
    # Assuming 'Class' is the target and selecting some relevant features
    selected_features = ['Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']
    
    # Check if the selected features exist in the dataset
    missing_features = [feature for feature in selected_features if feature not in d.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return "Error: Some selected features are missing in the dataset."
    
    # Extracting the features and target
    c = d.copy()
    x_train = c.loc[:350, selected_features]
    y_train = c.loc[:350, "Class"]
    x_test = c.loc[350:, selected_features]
    y_test = c.loc[350:, "Class"]

    knn_accuracy = KNN(x_train, y_train, x_test, y_test)
    naiveb_accuracy = NaiveB(x_train, y_train, x_test, y_test)
    decision_tree_accuracy = DecisionTree(x_train, y_train, x_test, y_test)
    
    # Render index.html with accuracy values
    return render_template('index.html', 
                           training_accuracy=training_accuracy, 
                           validation_accuracy=validation_accuracy,
                           knn_accuracy=knn_accuracy,
                           naiveb_accuracy=naiveb_accuracy,
                           decision_tree_accuracy=decision_tree_accuracy)

                           

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Ensure the uploads directory exists
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Get the uploaded file
        f = request.files['file']
        
        # Define the file path to save the uploaded file
        filepath = os.path.join(upload_folder, secure_filename(f.filename))
        
        # Save the file to the specified path
        f.save(filepath)
        
        # Get prediction result
        prediction = get_result(filepath)
        result = get_class_name(prediction)
        
        return result

if __name__ == '__main__':
    app.run(debug=True)
