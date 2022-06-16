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

    # Todo: normalize the data = each attribute value / max possible value of this attribute
    min_max_scaler = preprocessing.MinMaxScaler()
    train_scaled = min_max_scaler.fit_transform(x_train)
    x_train = pd.DataFrame(train_scaled)
    test_scaled = min_max_scaler.fit_transform(x_test)
    x_test = pd.DataFrame(test_scaled)

    # Todo: SVM classification
    clf = svm.SVC()
    # fit the model to the data
    clf.fit(np.array(x_train), np.array(y_train))
    # run the model to independent test data
    svm_labels_pred = clf.predict(np.array(x_test))

    # Todo: For the SVM classifier, try different Kernel functions and keep the most promising.


    # Todo: Evaluation
    conf_matrix = confusion_matrix(y_test, svm_labels_pred)
    print(conf_matrix)


    return 0


if __name__ == "__main__":
    main()
