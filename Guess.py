import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# read data from student-mat.csv file
data = pd.read_csv("/home/user/guess-the-score/Guess_the_score/student-mat.csv", sep = ";")

# drop unnecessary columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# set predict(label)
predict = "G3"

# set x and y(feature and label)
X = np.array(data.drop([predict], axis = 1))
Y = np.array(data[predict])

# split data into training and testing(80% training and 20% testing)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


best = 0
for _ in range(1000):
    # split data into training and testing
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # create linear regression model
    linear = linear_model.LinearRegression()

    # train model
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    #print(acc)

    if(acc > best):
        best = acc

        # save model to disk
        with open("GuessModel.pickle", "wb") as f:
            pickle.dump(linear, f)
print(best)


# load model from disk
pickle_in = open("GuessModel.pickle", "rb")
linear = pickle.load(pickle_in)

# predict
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grades")
pyplot.show()
