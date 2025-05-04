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
from sklearn import svm
import math
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

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

def SVM(x_training, y_training, x_testing, y_testing):
    print("\nSupport Vector Machine")
    clf = svm.SVC(gamma='auto')
    clf.fit(x_training, y_training)
    y_predict = clf.predict(x_testing)
    cm = confusion_matrix(y_testing, y_predict)
    accuracy = sum(cm.diagonal()) / len(x_testing) * 100
    return accuracy

@app.route('/')
def index():
    # Call ML algorithms and get accuracy
    d = pd.read_csv("C:/Users/CK STUDY CLASSES/Desktop/Brain Tumor Detection Accuracy/Brain_Tumer.csv")
    c = d.copy()
    x_train = c.loc[:350, : "mitosis"]
    y_train = c.loc[:350, "class"]
    x_test = c.loc[350:, : "mitosis"]
    y_test = c.loc[350:, "class"]

    knn_accuracy = KNN(x_train, y_train, x_test, y_test)
    naiveb_accuracy = NaiveB(x_train, y_train, x_test, y_test)
    decision_tree_accuracy = DecisionTree(x_train, y_train, x_test, y_test)
    svm_accuracy = SVM(x_train, y_train, x_test, y_test)

    # Render index.html with accuracy values
    return render_template('index.html', 
                           training_accuracy=training_accuracy, 
                           validation_accuracy=validation_accuracy,
                           knn_accuracy=knn_accuracy,
                           naiveb_accuracy=naiveb_accuracy,
                           decision_tree_accuracy=decision_tree_accuracy,
                           svm_accuracy=svm_accuracy)

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filepath = os.path.join('uploads', secure_filename(f.filename))
        f.save(filepath)
        prediction = get_result(filepath)
        result = get_class_name(prediction)
        return result

if __name__ == '__main__':
    app.run(debug=True)
