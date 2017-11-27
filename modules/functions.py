# Python standard library
import csv


# Scipy
import pandas as pd
import numpy as np
import sklearn

# Program-specific
import constants




def LogLoss(predicted, actual):
    '''
    Log loss, or cross-entropy error
    '''

    return -(np.sum(np.log(predicted[actual == 1])) + np.sum(np.log(1-predicted[actual == 0])))/len(actual)



def LoadTrainData(aug = True, mix = False):
    '''
    Loads the training data.
    Will load the augmented data as well if `aug == True`.
    '''

    df_train_raw = pd.read_csv(constants.train_raw_file_path, sep = ',', header = 0, index_col = 0)

    if aug:
        df_train_augmented = pd.read_csv(constants.train_aug_file_path, sep = ',', header = 0, index_col = 0)
        df_train = pd.concat([df_train_raw, df_train_augmented])

    else:
        df_train = df_train_raw


    if mix:
        df_train = sklearn.utils.shuffle(df_train)

    return df_train

def FillMissing(df, file_path):
    # Fill NA values w/ file values


    df_indices = df.index.values

    with open(file_path, 'r') as file_handle:
        reader = csv.reader(file_handle, delimiter = ',')

        header = next(reader)
        feature = header[1]

        dictt = {}
        for row in reader:
            if row[0] in df_indices:
                dictt[row[0]] = float(row[1])

    df.loc[list(dictt.keys()), feature] = list(dictt.values())


def LoadTestData(mix = False):
    '''
    Loads the testing data.
    '''

    df_test = pd.read_csv(constants.test_raw_file_path, sep = ',', header = 0, index_col = 0)

    if mix:
        df_test = sklearn.utils.shuffle(df_test)

    return df_test


def AvgLogLoss(predictions, labels):
    '''
    Calculate the average log-loss
    '''
    log_loss = 0

    log_loss += np.dot(-np.log(predictions[labels == 1]))

    log_loss += np.dot(-np.log(1-predictions[labels == 0]))

    log_loss /= (1.*len(predictions))

    return log_loss
