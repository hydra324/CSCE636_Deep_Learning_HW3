import numpy as np
import os

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 5.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx

def prepare_data(data_dir, train_filename, test_filename):
    """
    Args:
        data_dir: a string, the path to the folder containing data files.
        train_filename: a string, the name of the file containing training data.
        test_filename: a string, the name of the file containing test data.

    Returns:
        train_X, train_y: Arrays of shape [n_train_samples, 256] and [n_train_samples,], data and labels of training set.
        valid_X, valid_y: Arrays of shape [n_valid_samples, 256] and [n_valid_samples,], data and labels of validation set.
        train_valid_X, train_valid_y: Arrays of shape [n_samples, 256] and [n_samples,], 
                                                                  concatenation of train_X and valid_X, train_y and valid_y.
        test_X, test_y: Arrays of shape [n_test_samples, 256] and [n_test_samples,], data and labels of test set.
    """
    # Read data for training.

    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(
        raw_data, labels, 2300)

    # Preprocess raw data to extract features
    train_X_all = raw_train
    valid_X_all = raw_valid
    # Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)

    # For binary case, only use data from '1' and '2'
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    # set lables to  1 and 0. Here convert label '2' to '0' which means we treat data '1' as postitive class.
    train_y[np.where(train_y == 2)] = 0
    valid_y[np.where(valid_y == 2)] = 0

    train_valid_X = np.concatenate((train_X, valid_X))
    train_valid_y = np.concatenate((train_y, valid_y))

    test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_y_all, test_idx = prepare_y(test_labels)
    test_X_all = test_data
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = 0
    test_X = test_X_all[test_idx]

    return train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y