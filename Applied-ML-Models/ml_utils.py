from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

def space_util():
    print("-----------------")
    print("\n") 



###############################################################################
#                       LIBRARY IMPORTS AND SETUP                             #
###############################################################################

## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
# import ppscore

## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble
# import imblearn

## for deep learning
# from tensorflow.keras import models, layers
# import minisom

## for explainer
# from lime import lime_tabular
# import shap

## for geospatial
# import folium
# import geopy


###############################################################################
##                      DATA ANALYSIS                                         #
###############################################################################

'''
Recognize whether a column is numerical or categorical.
:parameter
    :param df: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
:return
    "cat" if the column is categorical and "num" otherwise
'''
def utils_recognize_type(df, col, max_cat=20):
    if (df[col].dtype == "O") | (df[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"


'''
Get a general overview of a dataframe.
:parameter
    :param df: dataframe - input data
    :param max_cat: num - mininum number of recognize column type
'''
def df_overview(df, max_cat=20, figsize=(10,5)):
    ## recognize column type
    dic_cols = {col:utils_recognize_type(df, col, max_cat=max_cat) for col in df.columns}

    printmd("**Shape of the Dataset:**")
    print(df.shape)
    space_util()

    printmd("**Number of rows and columns in the Dataset:**")
    print(df.keys())
    space_util()    

    printmd("**Basic information of the Dataset:**")
    print(df.info())
    space_util()

    printmd("**Percentage null values in the Dataset:**")
    print(((df.isnull().sum()/len(df))*100).sort_values(ascending=False))
    space_util()

    printmd("**Total number of duplicated rows in the Dataset:**")
    print(df.duplicated().sum())
    space_util()

    printmd("**Categorical columns in the Dataset:**")
    catCols = df.select_dtypes("object").columns
    catCols= list(set(catCols))
    print(catCols)
    space_util()


    printmd("**Unique values across categorical columns in the Dataset:**")
    def uniq_vals(col):
        print("Unique values in the column: ", col)
        print(df[col].value_counts().to_dict())
        space_util()

    _ = [uniq_vals(col) for col in catCols]
    space_util()


    printmd("**Visual representation of the Dataset:**")
    ## recognize column type
    dic_cols = {col:utils_recognize_type(df, col, max_cat=max_cat) for col in df.columns}

    ## plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = df.isnull()
    for k,v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Dataset Overview')
    #plt.setp(plt.xticks()[1], rotation=0)
    plt.show()
    
    ## add legend
    print("\033[1;37;40m Categorical \033[m", "\033[1;30;41m Numerical \033[m", "\033[1;30;47m NaN \033[m")


###############################################################################
#                       PREPROCESSING                                         #
###############################################################################
'''
Split the dataframe into train / test
'''
def df_partitioning(df, y, test_size=0.3, shuffle=False):
    df_train, df_test = model_selection.train_test_split(df, test_size=test_size, shuffle=shuffle) 
    print("X_train shape:", df_train.drop(y, axis=1).shape, "| X_test shape:", df_test.drop(y, axis=1).shape)
    print("y_train mean:", round(np.mean(df_train[y]),2), "| y_test mean:", round(np.mean(df_test[y]),2))
    print(df_train.shape[1], "features:", df_train.drop(y, axis=1).columns.to_list())
    return df_train, df_test


'''
Replace Na with a specific value or mean for numerical and mode for categorical. 
'''
def fill_na(df, col, value=None):
    if value is None:
        value = df[col].mean() if utils_recognize_type(df, col) == "num" else df[col].mode().iloc[0]
        print("--- Replacing Nas with:", value, "---")
        df[col] = df[col].fillna(value)
        return df, value
    else:
        print("--- Replacing Nas with:", value, "---")
        df[col] = df[col].fillna(value)
        return df


'''
Transforms a categorical column into dummy columns
:parameter
    :param df: dataframe - feature matrix df
    :param x: str - column name
    :param dropx: logic - whether the x column should be dropped
:return
    df with dummy columns added
'''
def add_dummies(df, x, dropx=False):
    df_dummy = pd.get_dummies(df[x], prefix=x, drop_first=True, dummy_na=False)
    df = pd.concat([df, df_dummy], axis=1)
    print( df.filter(like=x, axis=1).head() )
    if dropx == True:
        df = df.drop(x, axis=1)
    return df


'''
Maps a categorical column with two different values to 0 and 1
:parameter
    :param df: dataframe - feature matrix df
    :col_list: list - list of columns to be mapped
    :val_list: list - list of values to be mapped to 0 and 1, list with only 2 values

'''
def add_mapping(df, col_list, val_list):

    # Defining the map function
    def binary_map(x):
        return x.map({val_list[0]: 0, val_list[1]: 1})

    # Applying the function to the housing list
    df[col_list] = df[col_list].apply(binary_map)

    return df



'''
To be updated
Scales features
:parameter
    :param df: dataframe - feature matrix df
    :col_list: list - list of columns to be scaled
'''
def scaling(df, col_list):
    scaler = preprocessing.MinMaxScaler()
    df[col_list] = scaler.fit_transform(df[col_list])
    return df


###############################################################################
#                       DATA-VISUALIZATION                                    #
###############################################################################


'''
Visualize corr matrix
:parameter
    :param df: dataframe - feature matrix df
    :figsize: tutple
'''
def corr_matrix(df, figsize = (16,10)):
    plt.figure(figsize = figsize)
    sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
    plt.show()

'''
Visualize scatter plot 
:parameter
    :param df: dataframe - feature matrix df
    :col_list: list - list of columns to be visualized
    :figsize: tutple
'''
def scatter_plot(df, col_list, figsize = (6,6)):
    plt.figure(figsize = figsize)
    plt.scatter(df[col_list[0]], df[col_list[1]])
    plt.show()
