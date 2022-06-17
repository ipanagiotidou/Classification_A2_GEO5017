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


def evaluation(y_test, class_pred_RF, num_pc):
    # 1st Metric: Confusion matrix
    conf_matrix = confusion_matrix(y_test, class_pred_RF)
    # 2nd Metric: Overall Accuracy
    overall_accuracy = sum(np.diagonal(conf_matrix)) / len(y_test)
    print("Overall Accuracy", overall_accuracy)
    # 3rd Metric: Mean per-class accuracy
    mA = (1 / 5) * np.sum(np.divide(np.diagonal(conf_matrix), num_pc))
    print("Mean per class Accuracy: ", mA)
    return conf_matrix, overall_accuracy, mA


def main():

    # MULTI-CLASS CLASSIFICATION

    # todo: calculate features
    df = calculate_features.load_and_calculate_features()

    # Todo: Feature Selection Strategies - Scatter Metrics
    # feeding each feature at a time calculate the covariance
    features = ["volume", "proj_area", "density_3d", "median_height", "area_3d", "density_2d"]
    categories = ['building', 'car', 'fence', 'pole', 'tree']

    while len(features) > 3:
        criteria = []
        print(features)
        for i in range(len(features)):
            # current feature set
            features_current = [feat for id, feat in enumerate(features) if id != i]
            # initialize an empty dataframe
            df0 = pd.DataFrame(0, index=features_current, columns=features_current)

            # mean of all classes
            mean_all_classes = df[features_current].mean()
            # initialize an empty Series
            steps_Sb = pd.Series(0, index=features_current)

            for cat in categories:
                # covariance per class
                df_cov_per_class = 0.2 * df.loc[df.label == cat, features_current].cov()
                df0 = df0.add(df_cov_per_class, fill_value=0)

                # mean per class
                mean_per_class = df.loc[df.label == cat, features_current].mean()
                steps_Sb += 0.2 * (mean_per_class - mean_all_classes) * (mean_per_class - mean_all_classes).transpose()


            traceSw = np.trace(df0.to_numpy())
            steps_Sb = steps_Sb.to_numpy()
            traceSb = steps_Sb.sum()

            # compute criterion J
            criterion_J = traceSb/traceSw
            criteria.append(criterion_J)

        index = np.argmin(criteria)
        del features[index]

    print(features)



    # Todo: Separate the dataset into training and test set randomly
    x_train,x_test,y_train,y_test=train_test_split(
            df[["volume", "proj_area", "density_3d", "median_height", "area_3d", "density_2d"]],
            df['label'],test_size=0.4
            )

    # Todo: Try different train-test ratio
    # test_size=0.3, test_size=0.2, test_size=0.1

    # Retrieve information about the number of data samples of each class separately
    df = pd.DataFrame(y_test).reset_index()
    df.columns = ['id', 'label']
    num_pc = df.groupby(by=['label']).count()
    num_pc = np.array(num_pc['id']) # returns an array with the number of samples per class in the training set

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
    conf_matrix, overall_accuracy, mA = evaluation(y_test, class_pred_RF, num_pc)



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
        conf_matrix, overall_accuracy, mA = evaluation(y_test, labels_pred, num_pc)
        # add the kernel name and the corresponding accuracy in a dictionary
        dict[kernels[i]] = overall_accuracy
        i += 1

    # choose the most promising kernel retrieving the key of the maximum value from the dictionary
    chosen_kernel_SVM = max(zip(dict.values(), dict.keys()))[1] # best kernel: rbf for the 6:4 ratio






    return 0


if __name__ == "__main__":
    main()
