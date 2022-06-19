# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

def boxplot(dataset, title):
    df = pd.DataFrame(dataset)
    dt = df.to_numpy()
    df = pd.DataFrame(dt, columns=['6:4', '7:3', '8:2', '9:1'])
    df.plot.box(grid='True')
    plt.xlabel("train-test ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(title)
    plt.axis([0, 5, 0.4, 0.9])
    plt.show()

def plot_main():

    # # --- --- --- --- BOX PLOT FOR RANDOM FOREST --- --- --- ---
    # Todo: The dataset for Random Forest
    data_RF = {'0.4': [0.75, 0.76, 0.8], '0.3': [0.85, 0.81, 0.75] , '0.02': [0.76, 0.87, 0.8], '0.1': [0.48, 0.56, 0.88]}
    # Todo: Boxplot for Random Forest
    title = "Random Forest\n Accuracy of the different train-test ratios"
    boxplot(data_RF, title)

    # --- --- --- --- BOX PLOT FOR SUPPORT VECTOR MACHINE --- --- --- ---
    # Todo: The dataset for SVM
    data_SVM = {'0.4': [0.71, 0.73, 0.79], '0.3': [0.79, 0.8, 0.73], '0.02': [0.76, 0.73, 0.69], '0.1': [0.64, 0.74, 0.74]}
    # # Todo: Boxplot for SVM
    title = "Support Vector Machine\n Accuracy of the different train-test ratios"
    boxplot(data_SVM, title)
