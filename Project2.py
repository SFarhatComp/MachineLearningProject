import numpy as np
import graphviz
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score , ConfusionMatrixDisplay, classification_report
from sklearn.neural_network import MLPClassifier


def iris():
    iris = datasets.load_iris()
    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    X, y = iris.data, iris.target
    # Printing the total DATA
    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)

    dtc.fit(X_train, y_train)
    # print(" THE X TRAIN IS : ")
    # print(X_train)  # (n_samples_train, n_features)

    # print(" THE X TEST IS : ")
    # print(X_test)   # (n_samples_test, n_features)
    # print(" THE Y TRAIN IS : ")
    # print(y_train)  # (n_samples_train,)

    # print(" THE Y TEST IS : ")
    # print(y_test)   # (n_samples_test,)

    # Now we need to test.

    y_prediction = dtc.predict(X_test)

    dot_data = tree.export_graphviz(dtc, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("IrisTree")

    print(classification_report(y_test, y_prediction))

    confusion_matrix_ = confusion_matrix(y_test, y_prediction)
    print("The Confusion matrix is : ")
    print(confusion_matrix_)

    ConfusionMatrixDisplay(confusion_matrix_, display_labels=[1, 2, 3 ]).plot()
    plt.show()




def MNIST():
    MNIST = datasets.load_digits()

    n_samples = len(MNIST.images)
    data = MNIST.images.reshape((n_samples, -1))

    y = MNIST.target
    dtc = tree.DecisionTreeClassifier(criterion="entropy")

    print("The data is")
    print(data)


    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=10)

    dtc.fit(X_train, y_train)
    tree.plot_tree(dtc)
    # Now we need to test.

    y_prediction = dtc.predict(X_test)

    print(classification_report(y_test, y_prediction))

    confusion_matrix_ = confusion_matrix(y_test, y_prediction)
    print("The Confusion matrix is : ")
    print(confusion_matrix_)

    ConfusionMatrixDisplay(confusion_matrix_, display_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).plot()
    plt.show()


def MNISTPerceptron():
    MNIST = datasets.load_digits()
    n_samples = len(MNIST.images)
    data = MNIST.images.reshape((n_samples, -1))
    y = MNIST.target
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=5, shuffle=False)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                        verbose=0, random_state=5, solver="sgd")
    mlp.fit(X_train, y_train)
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


    ConfusionMatrixDisplay(confusion_matrix_, display_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).plot()
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



def IrisPerceptron():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=5)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000,
                        solver="lbfgs", random_state=5)
    mlp.fit(X_train, y_train)
    y_Prediction = mlp.predict(X_test)
    
    print("The acuracy function for the MLP")
    print(accuracy_score(y_test, y_Prediction))

    # print("The classification report is  for the MLP")
    # print(classification_report(y_test, y_Prediction))

    
    
    # PreMacro = precision_score(y_test, y_Prediction, average='macro')
    # PreMicro = precision_score(y_test, y_Prediction, average='micro')
    # RecallMacro = recall_score(y_test, y_Prediction, average='macro')
    # RecallMicro = recall_score(y_test, y_Prediction, average='micro')
    
    
    # print(f"PreMacro : {PreMacro}")
    # print(f"PreMicro : {PreMicro}")
    # print(f"RecallMacro : {RecallMacro}")
    # print(f"RecallMicro : {RecallMicro}")

    
    
    confusion_matrix_ = confusion_matrix(y_test, y_Prediction)
    print("The Confusion matrix is : ")
    print(confusion_matrix_)
    
    ConfusionMatrixDisplay(confusion_matrix_, display_labels=[1, 2]).plot()
    plt.show()
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

    


def main():

    iris()
    #MNIST()
    #MNISTPerceptron()
    #IrisPerceptron()


if __name__ == "__main__":
    main()
