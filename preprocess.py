import logging
import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from config import DATA_FILE, LOG_FILE, TEST_SIZE
from utils import save_data_to_pkl, stratified_split_data, profile

# # log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# @profile
def from_list_to_str(series):
    """Function to transform series of list(str) to series of string
    param: series:list(str)
    return: new series: str
    """
    return ' '.join([words for words in series])


# @profile
def load_data_transform_to_set(filename):
    """
    Read from pkl file,transform from dict(str:list(str),str:str)
    a normal dataset with labels, without nans and duplicates
    :param filename: string path to pkl file with saved json data
    :return: pandas DataFrame, columns=[paragraphs:str, labels:int]
                                (1 and 0 - recipe and instructions)
    """
    # load data
    if type(filename) == str:
        df = pd.DataFrame(pd.read_pickle(filename))
        # transform recipe to array
        recipe_col = df["Recipe"].apply(from_list_to_str).to_numpy()
    else:
        df = pd.DataFrame([filename])  # here we got a list(dict())
        recipe_col = df["Recipe"].to_numpy()

    recipe = recipe_col.reshape(-1, 1)

    # and give a label 1
    recipe = np.hstack((recipe, np.ones(len(recipe), int).reshape(-1, 1)))
    logging.info(f'Recipe transformed to array and have a label 1 with shape {recipe.shape}')

    # transform instructions to array and give a label 0
    instr_col = df["INSTRUCTIONS"].str.split('\n\n').to_numpy()
    instr = np.concatenate(instr_col).reshape(-1, 1)
    instr = np.hstack((instr, np.zeros(len(instr), int).reshape(-1, 1)))
    logging.info(f'INSTRUCTIONS transformed to array and have a label 0 with shape {instr.shape}')

    # forming a full data array with labels
    data = np.concatenate((instr, recipe), axis=0)
    logging.info('Shape  of all paragraphs data matrix ' + str(data.shape))

    # remove duplicates
    unique = np.unique(data.astype(str), axis=0)
    logging.info(f'Shape without duplicates {unique.shape}')

    # #remove empty string rows(from table)
    unique = np.delete(unique.astype(str), np.where(unique == ''), axis=0)
    logging.info(f'Shape without empty string rows {unique.shape}')

    return pd.DataFrame(unique, columns=['paragraph', 'label'])


@profile
def preprocess_clean_data(df, name_to_save):
    """
    The function replace numbers with the tag, lemmatize, counts prons,
    counts dots, counts numbers and removes stopwords
    saves as pkl file
    :param name_to_save: str
    :param  df: ndArray
    :return data:pandas df, path_to_data:str
    """

    # transform to pandas -> easy to clean
    data = pd.DataFrame(df, columns=['paragraph', 'label'])
    data['label'] = data.label.apply(int)  # this line (convert list --> int)

    path_to_data = save_data_to_pkl(data, f'new_{name_to_save}_data_clean.pkl')

    logging.info(f'Clean data is in {path_to_data}')
    logging.info(
        f'The proportion of target variable\n{round(data.label.value_counts() / len(data) * 100, 2)}')
    return path_to_data


@profile
def sent2vec(texts, max_sequence_length, vocab_size):
    """
    Create a union train set vocabulary and turn text in set
    into  padded sequences (word --> num )
    :param vocab_size: int lemmas count
    :param max_sequence_length: int max count words in series sentences
    :param texts: series of prepared strings
    :return ndArray with transformed series of text to int
            with 0-padding up to max_sequence_length
    """
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)

    # Turn text into  padded sequences (word --> num )
    text_sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(text_sequences, maxlen=max_sequence_length, padding="post", value=0)


@profile
def main_preprocess(filename=DATA_FILE):
    """
    Function load the data after scrapping from pkl file
    transform to data set with column 'paragraph' and 'label',
    split stratified on label  and preprocess separately train and test sets
    add new columns with additional features and save to 2 pkl files in ../data folder
    :param filename : str / default config.DATA_FILE
    :return void
    """
    # load data
    filename = os.getcwd() + '/data/' + filename  # ../data/recipes.pkl'
    logging.info(f'Data set is loading from {filename}')
    df = load_data_transform_to_set(filename)  # pd.DataFrame(unique,columns=['paragraph', 'label'])

    text = pd.DataFrame(df['paragraph'])
    label = pd.DataFrame(df['label']).astype(int)

    # data stratified split test_size = 0.2 default it is in config
    train_dataset, test_dataset = stratified_split_data(text, label, TEST_SIZE)  # data/train_data_clean.pkl

    # preprocessing for train and test sets
    preprocess_clean_data(train_dataset.as_numpy_iterator(), 'train')
    preprocess_clean_data(test_dataset.as_numpy_iterator(), 'test')
    logging.info('If you want to continue run model_train.py\nYour data is in the ../data folder')
    sys.exit('If you want to continue run model_train.py')


if __name__ == '__main__':
    main_preprocess()
