import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# read data from student-mat.csv file
data = pd.read_csv("/home/user/guess-the-score/student-mat.csv", sep=";")

# drop unnecessary columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# set predict(label)
predict = "G3"

# set x and y(feature and label)
X = np.array(data.drop([predict], axis = 1))
Y = np.array(data[predict])

# split data into training and testing(80% training and 20% testing)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# create linear regression model
linear = linear_model.LinearRegression()

# train model
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

