# Todo: Separate the dataset into training and test set randomly for each class separately
categories = ['building', 'car', 'fence', 'pole', 'tree']
x_train, x_test, y_train, y_test = [], [], [], []
df_x_train = pd.DataFrame(columns=["volume", "proj_area", "density_3d", "median_height", "area_3d", "density_3d"])
df_x_test = pd.DataFrame(columns=['label'])  # set the column names
df_y_train = pd.DataFrame(columns=["volume", "proj_area", "density_3d", "median_height", "area_3d", "density_3d"])
df_y_test = pd.DataFrame(columns=['label'])  # set the column names
for cat in categories:
    # print("\n", cat, "\n")
    x_train_cat, x_test_cat, y_train_cat, y_test_cat = train_test_split(
        df.loc[df.label == cat, ["volume", "proj_area", "density_3d", "median_height", "area_3d", "density_3d"]],
        df.loc[df.label == cat, ['label']], test_size=0.4
    )

    # df_x_train = pd.concat([df_x_train, x_train_cat], sort=False)
    df_x_test = pd.concat([df_x_test, x_test_cat], sort=False)
    # df_y_train = pd.concat([df_y_train, y_train_cat], sort=False, ignore_index=False)
    # df_y_test = pd.concat([df_y_test, y_test_cat], sort=False, ignore_index=True)

print("df_x_train: ", "\n", df_x_train, "\n")
print("df_x_test: ", "\n", df_x_test, "\n")
# print("df_y_train: ", "\n", df_y_train, "\n")
# print("df_y_test: ", "\n", df_y_test, "\n")


# NOTES - useful code
# print("x_train", "\n", x_train_cat, "\n")
# print("x_test", "\n", x_test_cat, "\n")
# print("y_train", "\n", y_train_cat, "\n")
# print("y_test", "\n", y_test_cat, "\n")

