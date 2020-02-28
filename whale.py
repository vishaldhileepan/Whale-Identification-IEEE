import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier 

#Gathering training data from data directory
train_list = os.listdir('data')
master_train = []
for i in range(len(train_list)):
    path = 'data/'+train_list[i]
    im = cv2.imread(path)
    resized = cv2.resize(im, (1,40000), interpolation = cv2.INTER_AREA)
    master_train.append(resized)
train_labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
master_train_1 = np.reshape(master_train,(len(master_train),-1))

#Gathering testing data from test directory
test_list = os.listdir('test')
master_test = []
for i in range(len(test_list)):
    path = 'test/'+test_list[i]
    im = cv2.imread(path)
    resized = cv2.resize(im, (1,40000), interpolation = cv2.INTER_AREA)
    master_test.append(resized)
test_labels = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
master_test1 = np.reshape(master_test,(len(master_test),-1))

#Instantiating a KNN class, training it, and predicting
learner = KNeighborsClassifier(2, weights='distance')
learner.fit(master_train_1,train_labels)
prediction = learner.predict(master_test1)

#Printing predictions
print(prediction)

#Finding the accuracy
count = 0
for i in range(len(test_labels)):
    if prediction[i]==test_labels[i]:
        count +=1
accuracy = count/len(test_labels)
print('Accuracy is', accuracy)
