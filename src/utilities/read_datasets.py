import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sys

def read_monk(dataset, rescale=False):

    """
    Reads the monks datasets, creates the labels for supervised classification and hide them to the classifier.
    Possibility to rescale the labels to [-1, +1] instead of [0, +1]

    Return monk dataset and labels (as numpy ndarrays)
    """

    # Read the .csv file containing the data. The first line contains the list of attributes. The data is assigned to a Pandas dataframe.
    col_names = ['class', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'Id']
    monk_dataset = pd.read_csv(f"../datasets/monks/{str(dataset)}", sep=" ", names=col_names)
    monk_dataset.set_index('Id', inplace=True)
    labels = monk_dataset.pop('class')

    # One-Hot Encoding - Transforming the dataset into a numpy array and applying One-Hot Encoding to the categorical variables.
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.int8)

    labels = labels.to_numpy()[:, np.newaxis]
    # if rescale is True, the class values are rescaled to [-1, 1] instead of [0, 1]
    if rescale:
        labels[labels == 0] = -1

    # Shuffle the dataset
    dataset = list(zip(monk_dataset, labels))
    np.random.shuffle(dataset)
    monk_dataset, labels = zip(*dataset)

    return monk_dataset, labels


def cup_data():
    """
    Reads CUP dataset, extracting training data, targets and test set
    """
    col_names = ['id', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'target_x', 'target_y', 'target_z']

    directory = "../datasets/cup/"
    file = "ML-CUP23-TR.csv"    

    # Read training data and targets and test set from csv files 
    tr_data = pd.read_csv(directory + file, sep=',', names=col_names[:11], skiprows=range(7), usecols=range(0, 11))
    tr_targets = pd.read_csv(directory + file, sep=',', names=col_names[11:], skiprows=range(7), usecols=range(11, 14))

    file = "ML-CUP23-TS.csv"
    cup_ts_data = pd.read_csv(directory + file, sep=',', names=col_names[:-3], skiprows=range(7), usecols=range(0, 11))

    # Transform dataframes into numpy arrays
    tr_data = tr_data.to_numpy(dtype=np.float32)
    tr_targets = tr_targets.to_numpy(dtype=np.float32)
    cup_ts_data = cup_ts_data.to_numpy(dtype=np.float32)

    # Shuffle the dataset
    dataset = list(zip(tr_data, tr_targets))
    np.random.shuffle(dataset)
    tr_data, tr_targets = zip(*dataset)

    return tr_data, tr_targets, cup_ts_data