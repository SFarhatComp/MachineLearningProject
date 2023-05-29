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
dtc = tree.DecisionTreeClassifier(criterion="entropy")
X, y = iris.data, iris.target
# Printing the total DATA
# print(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

#Splitting the data

dtc.fit(X_train, y_train)


#training the data


y_prediction = dtc.predict(X_test)


#prediciting the data
dot_data = tree.export_graphviz(dtc, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("IrisTree")


#rendering 
print(classification_report(y_test, y_prediction))

confusion_matrix_ = confusion_matrix(y_test, y_prediction)
print("The Confusion matrix is : ")
print(confusion_matrix_)

ConfusionMatrixDisplay(confusion_matrix_, display_labels=[1, 2, 3]).plot()
plt.show()


#confusion amtrix