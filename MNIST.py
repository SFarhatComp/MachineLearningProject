import numpy as np
import graphviz
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.neural_network import MLPClassifier

MNIST = datasets.load_digits()
n_samples = len(MNIST.images)
data = MNIST.images.reshape((n_samples, -1))

y = MNIST.target
dtc = tree.DecisionTreeClassifier(criterion="entropy")

print("The data is")
print(data)

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=10)


#trainign
dtc.fit(X_train, y_train)
tree.plot_tree(dtc)
# Now we need to test.

y_prediction = dtc.predict(X_test)

print(classification_report(y_test, y_prediction))

confusion_matrix_ = confusion_matrix(y_test, y_prediction)
print("The Confusion matrix is : ")
print(confusion_matrix_)

ConfusionMatrixDisplay(confusion_matrix_, display_labels=[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).plot()
plt.show()
