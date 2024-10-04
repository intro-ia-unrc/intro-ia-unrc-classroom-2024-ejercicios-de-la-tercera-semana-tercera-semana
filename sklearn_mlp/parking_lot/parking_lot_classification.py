import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


input_dir = './clf-data'
categories = ['empty', 'non_empty']

data = []
labels = []

for category in categories:
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        #TODO add the corresponding label to labels

data = np.asarray(data)
labels = np.asarray(labels)

#TODO split data intro training and testing sets

#TODO create an MLP classifier and train it on the training data

#TODO measure performance, and print out confussion matrix


