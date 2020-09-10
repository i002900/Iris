## Another implementation of Gaussian Naive Bayes for classification
##In this example, we are using the Iris dataset for determination the specific species of Iris which has the following columns:
##1. Sepal length
##2. Sepal width
##3. Petal length
##4. Petal width
##5. Species name
## The iris dataset is sourced from the seaborn library

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

iris = sns.load_dataset('iris')                #obtain the data set using the in-built function.
# this produces an arry of 150 rows (i.e. samples) and 5 columns 
X_array = iris.drop('species' , axis=1)  # this drops the species column and gives an array of 4 columns representing the features
y_labels = iris['species']


# Split this into a training and test data set
X_train , X_test , y_train , y_test = train_test_split(X_array , y_labels , train_size = 0.5, random_state = 0)

gnb = GaussianNB()  #instantiate the model from the class

#Train and the training set and predict on the test data set
y_model = gnb.fit(X_train, y_train).predict(X_test)


accuracy = accuracy_score(y_model , y_test)
print(f"The accuracy score of this model is : {accuracy}")
print(f"Out of a total of {X_test.shape[0]} species there were {(y_model != y_test).sum()} incorrectly classified")


#Convert the label series object to a numpy array for printing
y_test_labels = np.array(y_test)

#Compare the model with the labels in the test data set:
 
y_test_labels = np.array(y_test)    
print("Predictions\t\t Actuals")
for i in range(0,len(y_test)):
    if y_model[i] != y_test_labels[i]:                         ##Print mis-matches
        print(f"{y_model[i]}\t\t {y_test_labels[i]}")
 

