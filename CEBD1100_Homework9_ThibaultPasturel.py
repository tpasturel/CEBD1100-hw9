from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import model_selection

iris = datasets.load_iris()

## Allocating 80% to train data and 20% to test
x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state = 100)

X = iris.data[:, :4]
y = iris.target


## Getting an overview of all the data
plt.scatter(X[:,0],X[:,1],X[:,2],X[:,3])
plt.show()


## Plot train with cluster centers
model = KMeans(3)
model.fit(x_train)

plt.scatter(x_train[:,0],x_train[:,1],x_train[:,2],x_train[:,3])
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=50, color="red"); 
plt.show()

