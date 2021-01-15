import argparse
import logging
import os
import pandas as pd
from get_one import get_one
from config import  VOCAB_SIZE, MODEL_NAME, LOG_FILE
from model_train import eval_on_one_page
from preprocess import from_list_to_str, load_data_transform_to_set, preprocess_clean_data, tfidf
from utils import print_json, profile
# log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@profile
def main_for_one_link():
    """
    The function accept an url as an argument.
    Collects the needed data from a single webpage, print it into console
    Transform Recipe part from list to string, clean, preprocess text,
    save into pkl file and splits data into nlp and meta sets.
    Evaluate the pretrained model on this data and print into log metrics
    :return void
    """
    parser = argparse.ArgumentParser(description='Print the recipe json from given link')
    parser.add_argument('link')
    args = parser.parse_args()

    dict_file = get_one(str(args.link).strip())
    print_json(str(args.link).strip(), dict_file)

    dict_file['Recipe'] = from_list_to_str(dict_file['Recipe'][0])
    dict_file['INSTRUCTIONS'] = dict_file['INSTRUCTIONS'][0]

    # transform to data set (funny tiny dataset;)
    df = load_data_transform_to_set(dict_file)
    text = df['paragraph']

    one_page_data_path = preprocess_clean_data(df, 'one_page')
    one_page_set_clean = pd.read_pickle(one_page_data_path)

    # sent to sequence only for  NLP TRAIN
    tf_idf_one_page = tfidf(text, VOCAB_SIZE)

    # for other features train
    X_meta_one_page = one_page_set_clean[
        ['sent_count', 'num_count', 'clean_paragraph_len', 'verb_count', 'contains_pron']]
    y_one_page = one_page_set_clean['label']
    logging.info(f'{os.getcwd()}{MODEL_NAME}')
    model = f'{os.getcwd()}{MODEL_NAME}'
    eval_on_one_page(tf_idf_one_page, X_meta_one_page, y_one_page, model, text)


if __name__ == '__main__':
    main_for_one_link()
