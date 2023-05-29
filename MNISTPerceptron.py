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

#reshaping the data

y = MNIST.target


#split
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=5, shuffle=False)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                    verbose=0, random_state=5, solver="sgd")

#trainign
mlp.fit(X_train, y_train)

#predicitng
y_Prediction = mlp.predict(X_test)
print("The acuracy function for the MLP")
print(accuracy_score(y_test, y_Prediction))

PreMacro = precision_score(y_test, y_Prediction, average='macro')
PreMicro = precision_score(y_test, y_Prediction, average='micro')
RecallMacro = recall_score(y_test, y_Prediction, average='macro')
RecallMicro = recall_score(y_test, y_Prediction, average='micro')


print(f"PreMacro : {PreMacro}")
print(f"PreMicro : {PreMicro}")
print(f"RecallMacro : {RecallMacro}")
print(f"RecallMicro : {RecallMicro}")
confusion_matrix_ = confusion_matrix(y_test, y_Prediction)
print("The Confusion matrix is : ")
print(confusion_matrix_)


ConfusionMatrixDisplay(confusion_matrix_, display_labels=[
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).plot()
plt.show()

# 10 is used since it represents a good compromise between variance and bias
Kfold_ = KFold(n_splits=10)

# The .split method is part of the KFOLD object and it is a generator that splitsdata into K  folds.
mlp2 = MLPClassifier(hidden_layer_sizes=(
    100,), verbose=0, random_state=5, shuffle=False, solver="sgd")
for i, j in Kfold_.split(data, y):
    X_train = data[i]
    X_test = data[j]
    y_train = y[i]
    y_test = y[j]
    mlp2.fit(X_train, y_train)


print("The score function for the KFOLD")
print(mlp2.score(X_test, y_test))
