import pandas as pd
import numpy as np


def get_gini_index(df, cols):
    """
    Args:
        df (DataFrame):Input data
        cols (list): List of features that need to be inspected

    Returns:
         List of Gini indices, one for each column in cols.
         The index is calculated as follows:

            GINI =  [ m / (m-1) ] * \sum {i=1,..,m} f_i ^2

                where m is the number of classes and f_i is the relative
                frequence of the class i
    """

    gini_list = []

    # for each feature that we want to inspect
    for col in cols:

        # count the relative frequencies
        relative_frequencies = np.array(list(df[col].value_counts())) / len(df)

        # calculate the normalizing factor ( m / (m-1) )
        norm_factor = len(relative_frequencies) / (len(relative_frequencies) - 1)
        # calculate the gini index
        gini_index = norm_factor * (1 - np.sum(relative_frequencies**2))

        gini_list.append(gini_index)

    return gini_list


def get_shannon_index(df, cols):
    """
    Args:
        df (DataFrame):Input data
        cols (list): List of features that need to be inspected

    Returns:
         List of Shannon indices, one for each column in cols.
         The index is calculated as follows:

            Shannon =  - [1 / ln m] * \sum {i=1,..,m} f_i ln f_i

                where m is the number of classes and f_i is the relative
                frequence of the class i
    """

    shannon_list = []

    # for each feature that we want to inspect
    for col in cols:

        # count the relative frequencies
        relative_frequencies = np.array(list(df[col].value_counts())) / len(df)

        # calculate the normalizing factor ( 1 / ln m )
        norm_factor = 1 / np.log(len(relative_frequencies))
        # calculate the shannon index
        shannon_index = - norm_factor * np.sum(relative_frequencies * np.log(relative_frequencies))

        shannon_list.append(shannon_index)

    return shannon_list


def get_simpson_index(df, cols):
    """
    Args:
        df (DataFrame):Input data
        cols (list): List of features that need to be inspected

    Returns:
         List of Simpson indices, one for each column in cols.
         The index is calculated as follows:

            Simpson  =  [1 / (m - 1)] * ( 1 / \sum {i=1,..,m} f_i ^ 2)

                where m is the number of classes and f_i is the relative
                frequence of the class i
    """

    simpson_list = []

    # for each feature that we want to inspect
    for col in cols:

        # count the relative frequencies
        relative_frequencies = np.array(list(df[col].value_counts())) / len(df)

        # calculate the normalizing factor ( 1 / m - 1)
        norm_factor = 1 / (len(relative_frequencies) - 1)
        # calculate the shannon index
        simpson_index = norm_factor * (1 / np.sum(relative_frequencies ** 2) - 1)

        simpson_list.append(simpson_index)

    return simpson_list

def get_imbalance_ratio(df, cols):
    """
    Args:
        df (DataFrame):Input data
        cols (list): List of features that need to be inspected

    Returns:
         List of imbalance ratios, one for each column in cols.
         The index is calculated as follows:

            IR  =  [1 / (m - 1)] * [ (1 / \sum {i=1,..,m} f_i ^ 2  ) - 1]

                where m is the number of classes and f_i is the relative
                frequence of the class i
    """

    ir_list = []

    # for each feature that we want to inspect
    for col in cols:

        # count the frequencies
        frequences = np.array(list(df[col].value_counts()))

        # calculate the shannon index
        ir_index = np.min(frequences) / np.max(frequences)

        ir_list.append(ir_index)

    return ir_list