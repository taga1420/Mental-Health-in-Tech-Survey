import pandas as pd
import numpy as np
from sklearn import svm, neighbors
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from keras import models
#from keras import layers
data = pd.read_csv('finalissimo.csv', sep = ',')

#####PRE-PROCESSING OF DATA#####
################################

#work_interfere Class - Remove samples with NaN vaue in the 
data.dropna(axis=0, subset = ['work_interfere'], inplace=True)

#Transform nominal variables to numeric:
listLabelsData = list(data)
for a in listLabelsData[1::]:

    typesOfLabels = data[a].unique()
    #print(typesOfLabels)
    numericalLabels = list(range(0, len(typesOfLabels)))
    data[a].replace(typesOfLabels, numericalLabels, inplace = True)

''' Count how many samples have for treatment or not
count = 0
counters = 0
print(targetVector.unique())
for i in targetVector:
    if (i==1):
        count = count + 1
    else:
        counters = counters + 1
print(count)
print(counters)
'''

#We have 612 who sought treatment and 352 who dont
#The data is not balance, remove 260 samples of treatment(2)

treatment = data.loc[(data.treatment == 1)]
non_treatment = data.loc[(data.treatment == 0)]

treatment = treatment.sample(frac=1)
treatment = treatment[0:352]

newData = pd.concat([treatment, non_treatment])
newData = newData.sample(frac=1)

newData.to_csv("newData.csv")

#CLASSIFICATION
#SVM
indexes = np.random.rand(len(newData)) < 0.7
train = newData[indexes]
test = newData[~indexes]

targetVector = newData.treatment #No - 0, Yes - 1

classifier1 = svm.SVC()
classifier1.fit(train, train.treatment)
predictions1 = classifier1.predict(test)

tn, fp, fn, tp = sk.metrics.confusion_matrix(test.treatment, predictions1).ravel()
accuracy = (tp + tn) / (tp + tn + fn + fp)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print('SVM')
print('Accuracy: ', accuracy, '\nSensitivity: ', sensitivity, '\nSpecificity: ', specificity)
print('\nConfusion Matrix:\n',sk.metrics.confusion_matrix(test.treatment, predictions1))


#RANDOM FOREST - Da overfit

clf = RandomForestClassifier(n_estimators = 200, oob_score = True, n_jobs = -1, random_state =50,
                             max_features = "auto", min_samples_leaf = 100)
clf.fit(train, train.treatment)
preds = clf.predict(test)
tn1, fp1, fn1, tp1 = sk.metrics.confusion_matrix(test.treatment, preds).ravel()
accuracy1 = (tp1 + tn1) / (tp1 + tn1 + fn1 + fp1)
sensitivity1 = tp1 / (tp1 + fn1)
specificity1 = tn1 / (tn1 + fp1)

print('\nRANDOM Forest')
print('Accuracy: ', accuracy1, '\nSensitivity: ', sensitivity1, '\nSpecificity: ', specificity1)
print('\nConfusion Matrix:\n',sk.metrics.confusion_matrix(test.treatment, preds))

#KNN

print('\nKNN')
listAccuracy = []
listNeighbors = []

for x in range(1, len(train)):
    clf = neighbors.KNeighborsClassifier(x)
    knn_model = clf.fit(train, train.treatment)
    preds_KNN = clf.predict(test)
    tn2, fp2, fn2, tp2 = sk.metrics.confusion_matrix(test.treatment, preds_KNN).ravel()
    accuracy2 = (tp2 + tn2) / (tp2 + tn2 + fn2 + fp2)
    sensitivity2 = tp2 / (tp2 + fn2)
    specificity2 = tn2 / (tn2 + fp2)

    listAccuracy.append(accuracy2)
    listNeighbors.append(x)

plt.figure(1)
plt.title('Accuracy vs Number of Nearest Neighbours')
plt.plot(listNeighbors, listAccuracy)
plt.xlabel('Number of Nearest Neigbours')
plt.ylabel('Accuracy')
plt.show()

print('Max Accuracy: ', max(listAccuracy), '\nNumber of Neighbours: ', listNeighbors[listAccuracy.index(max(listAccuracy))])

clf = neighbors.KNeighborsClassifier(listNeighbors[listAccuracy.index(max(listAccuracy))])
knn_model = clf.fit(train, train.treatment)
preds_KNN = clf.predict(test)
tn2, fp2, fn2, tp2 = sk.metrics.confusion_matrix(test.treatment, preds_KNN).ravel()
accuracy2 = (tp2 + tn2) / (tp2 + tn2 + fn2 + fp2)
sensitivity2 = tp2 / (tp2 + fn2)
specificity2 = tn2 / (tn2 + fp2)

print('\nFinal Values: \nAccuracy: ', accuracy2, '\nSensitivity: ', sensitivity2, '\nSpecificity: ', specificity2)
print('\nConfusion Matrix:\n',sk.metrics.confusion_matrix(test.treatment, preds_KNN))


#NEURAL NETWORK
'''
# Start neural network
network = models.Sequential()
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation='relu'))
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation='relu'))
# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation='sigmoid'))
# Compile neural network
network.compile(loss='binary_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric
# Train neural network
history = network.fit(train, # Features
                      train.treatment, # Target vector
                      epochs=3, # Number of epochs
                      verbose=1, # Print description after each epoch
                      batch_size=100, # Number of observations per batch
                      validation_data=(test, test.treatment)) # Data for evaluation
'''