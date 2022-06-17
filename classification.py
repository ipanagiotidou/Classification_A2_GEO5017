# Import modules
import calculate_features

# Import libraries
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def main():

    # MULTI-CLASS CLASSIFICATION

    # todo: calculate features
    df = calculate_features.load_and_calculate_features()

    # Todo: Separate the dataset into training and test set randomly
    x_train,x_test,y_train,y_test=train_test_split(df[["volume", "proj_area", "density_3d", "median_height", "area_3d", "density_3d"]],
                                                   df['label'],test_size=0.4)
    # Todo: Try different train-test ratio
    # ...


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


    ### --- --- --- IMPLEMENTATION OF RANFOM FOREST CLASSIFIER --- --- ---

    # Todo: Random Forest Classifier
    # define the model
    model = RandomForestClassifier()
    # fit the model on the whole dataset
    model.fit(np.array(x_train), np.array(y_train))
    # make predictions
    class_pred_RF = model.predict(np.array(x_test))


    # Todo: Evaluation of Random Forest
    print("--- --- Random Forest Evaluation: --- --- ")
    # 1st Metric: Confusion matrix
    conf_matrix = confusion_matrix(y_test, class_pred_RF)
    # 2nd Metric: Overall Accuracy
    overall_accuracy = sum(np.diagonal(conf_matrix)) / len(y_test)
    print("Overall Accuracy", overall_accuracy)
    # 3rd Metric: Mean per-class accuracy
    mA = (1 / 5) * np.sum(np.divide(np.diagonal(conf_matrix), num_pc))
    print("Mean per class Accuracy: ", mA)

    ### --- --- --- IMPLEMENTATION OF SVM CLASSIFIER --- --- ---

    # Todo: SVM classification, try different kernels and keep the most promising.
    # call the different kernels
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    labels_SVM_kernels = []
    for i in range(len(kernels)):
        clf = svm.SVC(kernel=kernels[i]) # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        clf.fit(np.array(x_train), np.array(y_train))
        svm_labels_pred = clf.predict(np.array(x_test))
        labels_SVM_kernels.append(svm_labels_pred)


    # Todo: Evaluation of SVM
    print("\n--- --- --- SVM Evaluation: --- --- ---")
    i = 0
    dict = {}
    for labels_pred in labels_SVM_kernels:
        print('\'', kernels[i], '\'', 'kernel')

        # 1st Metric: Confusion matrix
        conf_matrix = confusion_matrix(y_test, labels_pred)
        # 2nd Metric: Overall Accuracy
        overall_accuracy = sum(np.diagonal(conf_matrix)) / len(y_test)
        print("Overall Accuracy", overall_accuracy)
        # 3rd Metric: Mean per-class accuracy
        mA = (1/5) * np.sum(np.divide(np.diagonal(conf_matrix), num_pc))
        print("Mean per class Accuracy: ", mA)

        # add the kernel name and the corresponding accuracy in a dictionary
        dict[kernels[i]] = overall_accuracy
        i += 1

    # # choose the most promising kernel retrieving the key of the maximum value from the dictionary
    chosen_kernel_SVM = max(zip(dict.values(), dict.keys()))[1] # rbf for the 6:4






    return 0


if __name__ == "__main__":
    main()
