from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.svm import SVC
import numpy as np

dataset = read_csv("android-games.csv", header=0)

#print(dataset.head(4))
#print(dataset.describe())

array = dataset.values
X1 = array[:,4:8]
#print(X1)
X2 = array[:,9:14]
#print(X2)
y = array[:,14]
y=y.astype('bool')
#print(y)
X3 = np.concatenate((X1, X2), axis=1) #putting two arrays together by row
model = SVC(gamma="auto")
model.fit(X3, y)
print(model.predict(X3))