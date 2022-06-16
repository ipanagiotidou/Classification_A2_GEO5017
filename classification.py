# Import modules
import calculate_features

# Import libraries
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

def main():

    # MULTI-CLASS CLASSIFICATION

    # todo: calculate features
    df = calculate_features.load_and_calculate_features()

    # Todo: Separate the dataset into training and test set randomly
    x_train,x_test,y_train,y_test=train_test_split(df[["density_3d", "height", "area_3d"]],df['label'],test_size=0.4)

    df = pd.DataFrame(y_test).reset_index()
    df.columns = ['id', 'label']
    num_pc = df.groupby(by=['label']).count()
    num_pc = np.array(num_pc['id'])

    # Todo: normalize the data = each attribute value / max possible value of this attribute
    min_max_scaler = preprocessing.MinMaxScaler()
    train_scaled = min_max_scaler.fit_transform(x_train)
    x_train = pd.DataFrame(train_scaled)
    test_scaled = min_max_scaler.fit_transform(x_test)
    x_test = pd.DataFrame(test_scaled)


    # Todo: SVM classification, try different kernels and keep the most promising.
    # todo: call the different kernels at once because every time the sample set changes
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    labels_SVM_kernels = []
    for k in kernels:
        svm_labels_pred = []
        clf = svm.SVC(kernel='linear') # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        clf.fit(np.array(x_train), np.array(y_train))
        svm_labels_pred = clf.predict(np.array(x_test))
        labels_SVM_kernels.append(svm_labels_pred)


    # Todo: Evaluation
    # Metric: Confusion Matrix
    for labels_pred in labels_SVM_kernels:
        conf_matrix = confusion_matrix(y_test, labels_pred) \
        # Metric: Overall Accuracy
        overall_accuracy = sum(np.diagonal(conf_matrix)) / len(y_test)
        print("Overall Accuracy", overall_accuracy)
        # Metric: Mean per-class accuracy
        mA = (1/5) * np.sum(np.divide(np.diagonal(conf_matrix), num_pc))
        print("Mean per class Accuracy: ", mA)







    return 0


if __name__ == "__main__":
    main()
