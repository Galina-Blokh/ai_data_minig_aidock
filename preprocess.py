import logging
import os
import re
import numpy as np
import pandas as pd
from config import DATA_FILE, DIGIT_RX, SYMBOL_RX, DOT_RX, LOG_FILE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def replace_numbers_str(series):
    """
    replace numbers expression (11 ; 11,00;  1111.99; 23-th 25-,1/2,¼) with tag ' zNUM ',
    . and : replace with ' zDOT '
    :param series: pandas series of text strings
    :return: series with replaced numbers
    """
    new_series1 = re.sub(DIGIT_RX, " zNUM ", series)
    new_series2 = re.sub(SYMBOL_RX, ' ', new_series1)
    new_series = re.sub(DOT_RX, " zDOT ", new_series2)
    return new_series


def count_paragraph_stats(series):
    """Count number of sentences in the paragraph
        :param series: pandas series of text strings
        :return: int count of sentences in the string (if no dots -> count = 1)"""
    sent_count = series.count(' zDOT ')
    if sent_count == 0:
        sent_count = 1
    return sent_count


def num_count(series):
    """Count number of sentences in the paragraph
        :param series: pandas series of text strings
        :return: int count of sentences in the string"""
    return series.count('zNUM')


def load_data_transform_to_set(filename):
    """
    Read from pkl file,transform from dict(str:list(str),str:str)
    a normal dataset with labels, without nans and duplicates
    :param filename: string path to pkl file with saved json data
    :return: pandas DataFrame, columns=[paragraphs:str, labels:int]
                                (1 and 0 - recipe and instructions)
    """
    df = pd.DataFrame(pd.read_pickle(filename))

    # transform recipe to array and give a label 1
    recipe_col = df["Recipe"].to_numpy()
    recipe = np.concatenate(recipe_col).reshape(-1, 1)
    recipe = np.hstack((recipe, np.ones(len(recipe), int).reshape(-1, 1)))

    # transform instructions to array and give a label 0
    instr_col = df["INSTRUCTIONS"].str.split('\n').to_numpy()
    instr = np.concatenate(instr_col).reshape(-1, 1)
    instr = np.hstack((instr, np.zeros(len(instr), int).reshape(-1, 1)))

    # forming a full data array with labels
    data = np.concatenate((instr, recipe))
    logging.info('shape all of paragraphs data matrix' + str(data.shape))

    # remove duplicates
    unique = np.unique(data, axis=0)
    logging.info('shape without duplicates'+ str(unique.shape))

    # remove empty string rows(from table)
    unique = np.delete(unique, np.where(unique == ''), axis=0)
    logging.info('shape without empty string rows'+str(unique.shape))
    return pd.DataFrame(unique, columns=['paragraph', 'label'])


if __name__ == '__main__':
    # load data
    filename = f'{os.getcwd()}/data/{DATA_FILE}'
    df = load_data_transform_to_set(filename)

    # transform to pandas -> easy to clean
    df['paragraph'] = df.paragraph.apply(replace_numbers_str)
    # df[1:4]

    # TODO remove stopwords, punctuation tags and number tags
    # TODO Tokenize
    # TODO Lematize
    # TODO Word embeddings
