import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore
import sys

def read_monk(dataset, rescale=False):
    """
    Reads the monks datasets and returns the dataset and the labels as numpy ndarrays.
    Possibility to rescale the labels to [-1, +1] instead of [0, +1]

    Return monk dataset and labels (as numpy ndarrays)
    """

    # Read the .csv file containing the data. The first line contains the list of attributes. The data is assigned to a Pandas dataframe.
    col_names = ['class', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'Id']
    path = f"../datasets/monks/{str(dataset)}"
    monk_dataset = pd.read_csv(path, sep=" ", names=col_names)
    monk_dataset.set_index('Id', inplace=True)
    labels = monk_dataset.pop('class')

    # One-Hot Encoding - Transforming the dataset into a numpy array and applying One-Hot Encoding to the categorical variables.
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.int8)

    labels = labels.to_numpy()[:, np.newaxis]
    # if rescale is True, the class values are rescaled to [-1, 1] instead of [0, 1]
    if rescale:
        labels[labels == 0] = -1

    # Shuffle the dataset
    monk_dataset, labels = shuffle(monk_dataset, labels)

    return monk_dataset, labels


def read_cup(normalize=False):
    """
    Reads CUP dataset, extracting training data, targets and test set
    """
    col_names = ['id', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'target_x', 'target_y', 'target_z']

    directory = "../datasets/cup/"
    file = "ML-CUP23-TR.csv"    

    # Read training data and targets and test set from csv files 
    tr_data = pd.read_csv(directory + file, sep=',', names=col_names[1:11], skiprows=range(7), usecols=range(1, 11))
    tr_targets = pd.read_csv(directory + file, sep=',', names=col_names[11:], skiprows=range(7), usecols=range(11, 14))


    file = "ML-CUP23-TS.csv"
    cup_ts_data = pd.read_csv(directory + file, sep=',', names=col_names[1:11], skiprows=range(7), usecols=range(1, 11))

    # Transform dataframes into numpy arrays
    tr_data = tr_data.to_numpy(dtype=np.float32)
    tr_targets = tr_targets.to_numpy(dtype=np.float32)
    cup_ts_data = cup_ts_data.to_numpy(dtype=np.float32)

    if normalize:
        # Normalize the data
        tr_data = zscore(tr_data)

    # Shuffle the dataset
    tr_data, tr_targets = shuffle(tr_data, tr_targets)

    return tr_data, tr_targets, cup_ts_data


def shuffle(inputs, targets):
    indexes = np.random.permutation(len(inputs))
    return inputs[indexes], targets[indexes]


def split_dataset(dataset_inputs, dataset_targets, validation_percentage, test=False, test_percentage=0):
    """
    Splits the dataset into training, validation and test set
    :param dataset_inputs: The inputs of the dataset
    :param dataset_targets: The targets of the dataset
    :param validation_set_size: The size of the validation set
    :param test_set_size: The size of the test set
    :return: The training, validation and test set
    """
    if validation_percentage < 0 or validation_percentage > 1:
        print("The validation percentage must be between 0 and 1")
        sys.exit(1)
    if test_percentage < 0 or test_percentage > 1:
        print("The test percentage must be between 0 and 1")
        sys.exit(1)
    if validation_percentage + test_percentage > 1:
        print("The sum of the validation and test percentage must be less than 1")
        sys.exit(1)
    if not test and test_percentage > 0:
        print("The test percentage must be 0 if test is False")
        sys.exit(1)
    elif validation_percentage + test_percentage >= 1:
        print("The sum of the validation and test percentage must be less than 1")
        sys.exit(1)

    # Split the dataset into training, validation and (optionally) test set
    dataset_length = len(dataset_inputs)
    validation_set_size = math.floor(dataset_length * validation_percentage)
    test_set_size = math.floor(dataset_length * test_percentage)
    training_set_size = dataset_length - validation_set_size - test_set_size

    training_set_inputs = dataset_inputs[0:training_set_size]
    training_set_targets = dataset_targets[0:training_set_size]
    
    if test:
        validation_set_inputs = dataset_inputs[training_set_size:training_set_size + validation_set_size]
        validation_set_targets = dataset_targets[training_set_size:training_set_size + validation_set_size]
        test_set_inputs = dataset_inputs[training_set_size + validation_set_size:]
        test_set_targets = dataset_targets[training_set_size + validation_set_size:]
        return training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, test_set_inputs, test_set_targets
    else:
        validation_set_inputs = dataset_inputs[training_set_size:]
        validation_set_targets = dataset_targets[training_set_size:]
        return training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets