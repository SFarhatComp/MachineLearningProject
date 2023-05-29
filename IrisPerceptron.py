import numpy as np
import graphviz
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
X, y = iris.data, iris.target

#split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=5)
mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000,
                    solver="lbfgs", random_state=5)

#trainign and predictiong
mlp.fit(X_train, y_train)
y_Prediction = mlp.predict(X_test)


#values
print("The acuracy function for the MLP")
print(accuracy_score(y_test, y_Prediction))


confusion_matrix_ = confusion_matrix(y_test, y_Prediction)
print("The Confusion matrix is : ")
print(confusion_matrix_)

ConfusionMatrixDisplay(confusion_matrix_, display_labels=[1, 2]).plot()
plt.show()

#KFOLD MLP
Kfold_ = KFold(n_splits=10)

mlp2 = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000,
                     solver="lbfgs", random_state=5)

for i, j in Kfold_.split(X, y):
    X_train = X[i]
    X_test = X[j]
    y_train = y[i]
    y_test = y[j]
    mlp2.fit(X_train, y_train)

print("The score function for the KFOLD")
print(mlp2.score(X_test, y_test))
