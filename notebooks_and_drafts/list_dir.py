import glob
import logging
import os
import pandas as pd
from config import VOCAB_SIZE, LOG_FILE, TEST_LINKS_FILE
from get_one import get_one
from preprocess import from_list_to_str, load_data_transform_to_set, preprocess_clean_data, tfidf, eval_on_one_page
from utils import print_json
# log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def run_list_dir():
    """
    To test several models on several links (takes all from /data folder)
    """
    f = open(os.pardir+'/data/'+TEST_LINKS_FILE, 'r')
    models_list = glob.glob(f'{os.pardir}/data/' + '*.h5')
    list_links = [l.strip() for l in f]
    for url in list_links:
        dict_file = get_one(url)
        print_json(url, dict_file)

        dict_file['Recipe'] = from_list_to_str(dict_file['Recipe'][0])
        dict_file['INSTRUCTIONS'] = dict_file['INSTRUCTIONS'][0]
        # transform to data set (funny tiny dataset;)
        df = load_data_transform_to_set(dict_file)
        text = df['paragraph']

        one_page_data_path = preprocess_clean_data(df, 'one_page')
        one_page_set_clean = pd.read_pickle(one_page_data_path)

        # sent to sequence only for  NLP set
        tf_idf_one_page = tfidf(text, VOCAB_SIZE)
        # for other features set
        X_meta_one_page = one_page_set_clean[
            ['sent_count', 'num_count', 'clean_paragraph_len', 'verb_count', 'contains_pron']]
        y_one_page = one_page_set_clean['label']
        for model in models_list:
            eval_on_one_page(tf_idf_one_page, X_meta_one_page, y_one_page, model, text)
        print('Evaluation is finished')


if __name__ == '__main__':
    run_list_dir()