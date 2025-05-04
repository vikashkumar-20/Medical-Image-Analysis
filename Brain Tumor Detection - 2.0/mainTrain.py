import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

# Directory containing the images
image_directory = 'datasets/'

# List all images in the respective directories
no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))

# Initialize empty lists to store images and labels
dataset = []
label = []

# Define input size for resizing images
INPUT_SIZE = 64

# Load no tumor images
for image_name in no_tumor_images:
    if image_name.endswith('.jpg'):
        image = cv2.imread(os.path.join(image_directory, 'no', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Load yes tumor images
for image_name in yes_tumor_images:
    if image_name.endswith('.jpg'):
        image = cv2.imread(os.path.join(image_directory, 'yes', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# Convert the dataset list to a NumPy array
dataset = np.array(dataset)
label = np.array(label)

# Split the dataset and labels into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# Normalize the training and testing data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Convert the training and testing labels to categorical format
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model and save training and validation accuracy
history = model.fit(x_train, y_train, 
                    batch_size=16, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(x_test, y_test), 
                    shuffle=False)

# Save the model
model.save('BrainTumor10EpochsCategorical.h5')




# Save accuracy values
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
np.save('training_accuracy.npy', training_accuracy)
np.save('validation_accuracy.npy', validation_accuracy)


# Save accuracy values
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
np.save('training_accuracy.npy', training_accuracy)
np.save('validation_accuracy.npy', validation_accuracy)