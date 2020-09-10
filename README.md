# Iris
This is very quick and simple demonstration of using the Gaussian Naive Bayes algorithm for determining the species of Iris flower based on specific features. The ML model is based on the Scikit learn Gaussian Naive Bayes algorithm. The Iris data set is sourced from the seaborn library. The dataset contains five columns: a set of four features and a fifth column identifying the species.

<img src = "https://github.com/i002900/Iris/blob/master/iris_dataset.JPG">

There are just 150 entries. These are split into a Test and Training data set. The splitting is done using a standard train_test_split function available in sklearn. The standard function without any parameters gives a 75/25 split. Given the very small sample size, this has the effect of using a very large number of entries as a training set. This will give a hundred % accuracy score! We therefore split the set 50/50 by using the parameter train_size = 0.5.

The model is trained using the four features provided: Sepal length, sepal width, petal length and petal width.

This quick implementation gives a prediction accuracy of 0.94666
<img src = "https://github.com/i002900/Iris/blob/master/iris_4.JPG">

As can be seen out of a total of 75 species in the remainder Test set, only 4 were classified incorrectly. The mis-matches are printed out as shown below:

<img src = "https://github.com/i002900/Iris/blob/master/iris_5.JPG">
