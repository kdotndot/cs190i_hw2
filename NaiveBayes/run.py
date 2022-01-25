import numpy as np
import matplotlib.pyplot as plt
# An example dataset generator
#from sklearn.datasets import make_blobs
import csv
import pandas as pd
def plot_decision_boundary(classifier, X_test, y_test):
    # create a mesh to plot in
    h = 0.02  # step size in mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    X_hypo = np.c_[xx.ravel().astype(np.float32),
                   yy.ravel().astype(np.float32)]
    zz = classifier.predict(X_hypo)
    zz = zz.reshape(xx.shape)
    
    plt.contourf(xx, yy, zz, cmap='RdBu', alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='RdBu', edgecolor="k")


points = []
labels = []
with open("Xtrain.csv") as file_name:
    points = np.loadtxt('Xtrain.csv', delimiter=",")
    labels = np.loadtxt('Ytrain.csv', delimiter=",")


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(points, labels)
# You can see the predicted probability of each class given X by predict_proba(X).
yprob = model.predict_proba(points)
yprob[-10:].round(2)
print(yprob)







