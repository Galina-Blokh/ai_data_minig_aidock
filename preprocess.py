import datetime
import logging
import os
import re
import string

import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# python3 -m spacy download en_core_web_sm
from utils import save_data_to_pkl, timeit

nlp = spacy.load('en_core_web_sm')
nlp.Defaults.stop_words |= {" f ", " s ", " etc"}
# import en_core_web_sm
stop_words = set([w.lower() for w in list(STOP_WORDS)])
from config import DATA_FILE, DIGIT_RX, SYMBOL_RX, DOT_RX, LOG_FILE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@timeit
def replace_numbers_str(series):
    """
    Replace numbers expression (11 ; 11,00;  1111.99; 23-th 25-,1/2,¼) with tag ' zNUM ',
    . and : replace with ' zDOT '
    :param series: pandas series of text strings
    :return: series with replaced numbers
    """
    new_series1 = re.sub(DIGIT_RX, " zNUM ", series)
    new_series2 = re.sub(SYMBOL_RX, ' ', new_series1)
    new_series = re.sub(DOT_RX, " zDOT ", new_series2)
    return new_series


@timeit
def lemmatiz(series):
    """Transform all words to lemma, add tag  -PRON-
    param: series:str
    return: series:str"""
    new_series = ' '.join([word.lemma_ for word in nlp(series)])
    return new_series


@timeit
def have_pron(series):
    """Give the answer is there a pron in the paragraph
    param:series:str
    return: int 1 or 0"""
    answer = 0
    if series.__contains__('-PRON-'):
        answer = 1
    return answer


@timeit
def remove_punctuation(series):
    """ Remove punctuation from each word, make every word to lower case
    params: series of strings
    return transformed series"""
    table = str.maketrans('', '', string.punctuation)
    tokens_punct = series.translate(table).lower()
    return tokens_punct


@timeit
def remove_stop_words(series):
    """Remove stopwords in the series:str and makes words lower case
    param:series:str
    return: new_series:str
        """
    new_series = ' '.join([word for word in series.split() if word not in stop_words])
    return new_series


@timeit
def count_paragraph_sentences(series):
    """Count number of sentences in the paragraph
        :param series: pandas series of text strings
        :return: int count of sentences in the string (if no dots -> count = 1)"""
    sent_count = series.count(' zdot ')
    if sent_count == 0:
        sent_count = 1
    return sent_count


@timeit
def num_count(series):
    """Count number of sentences in the paragraph
        :param series: pandas series of text strings
        :return: int count of sentences in the string"""
    return series.count('znum')


@timeit
def count_words(series):
    """ Count words in each string (paragraph)
    without dots and numbers
    params: series of strings
    returns: new series:int"""

    clean_sent_len = len([word for word in series.split()
                          if word not in ['znum', 'zdot']])
    return clean_sent_len


@timeit
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
    logging.info('Recipe transformed to array and give a label 1')

    # transform instructions to array and give a label 0
    instr_col = df["INSTRUCTIONS"].str.split('\n\n').to_numpy()
    instr = np.concatenate(instr_col).reshape(-1, 1)
    instr = np.hstack((instr, np.zeros(len(instr), int).reshape(-1, 1)))
    logging.info('INSTRUCTIONS transformed to array and give a label 0')

    # forming a full data array with labels
    data = np.concatenate((instr, recipe))
    logging.info('Shape  of all paragraphs data matrix ' + str(data.shape))

    # remove duplicates
    unique = np.unique(data, axis=0)
    logging.info('Shape without duplicates ' + str(unique.shape))

    # remove empty string rows(from table)
    unique = np.delete(unique, np.where(unique == ''), axis=0)
    logging.info('Shape without empty string rows ' + str(unique.shape))
    return pd.DataFrame(unique, columns=['paragraph', 'label'])


@timeit
def main_preprocess():
    # load data
    filename = f'{os.getcwd()}/data/{DATA_FILE}'
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    logging.info('Data set is loading from ' + str(filename))
    df = load_data_transform_to_set(filename)

    # transform to pandas -> easy to clean
    df['paragraph'] = df.paragraph.apply(replace_numbers_str)
    logging.info('Numbers are replaced by tag')

    # TODO preprocess.py", line 156, in main
    # df['paragraph'] = df.paragraph.apply(replace_numbers_str)
    # AttributeError: 'NoneType' object has no attribute 'paragraph'

    # this one takes an eternity (lemmatiz)
    df['lemmatiz'] = df.paragraph.apply(lemmatiz)
    logging.info('Lemmatization is done')

    df['contains_pron'] = df.lemmatiz.apply(have_pron)
    logging.info('column contains_pron created')
    df['tokens_punct'] = df.lemmatiz.apply(remove_punctuation)

    df['remove_stop_words'] = df.tokens_punct.apply(remove_stop_words)
    logging.info('column remove_stop_words is created')

    df['sent_count'] = df.remove_stop_words.apply(count_paragraph_sentences)
    logging.info('column sent_count is created')

    df['num_count'] = df.remove_stop_words.apply(num_count)
    logging.info('column num_count is created')

    df['clean_paragraph_len'] = df.remove_stop_words.apply(count_words)
    logging.info('column clean_paragraph_len is created')

    df['not_clean_paragraph_len'] = df.paragraph.apply(count_words)
    logging.info('column num_count is created')

    data = df[['remove_stop_words', 'sent_count', 'num_count', 'clean_paragraph_len', 'contains_pron', 'label']]
    path_to_data = save_data_to_pkl(df, 'data_clean.pkl')

    stop_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    total_time = (datetime.datetime.strptime(stop_time, '%H:%M:%S.%f') -
                  datetime.datetime.strptime(start_time, '%H:%M:%S.%f'))
    print(f'{total_time} Clean data is in {path_to_data}')


if __name__ == '__main__':
    main_preprocess()

    # TODO Tokenizer
    # TODO Word2Vec
    # TODO RNN/LSTM
    # TODO chain everything in for_one_link_run.py
