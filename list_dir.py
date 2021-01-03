import glob
import logging
import os
import pandas as pd
import extract_one_recipe
from config import MAX_SEQ_LEN, VOCAB_SIZE, LOG_FILE
from preprocess import from_list_to_str, load_data_transform_to_set, preprocess_clean_data, sent2vec
from run_tensorflow import eval_on_one_page
from utils import print_json
# log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':

    f = open(f'{os.getcwd()}/data/test_links.txt', 'r')
    models_list = glob.glob(f'{os.getcwd()}/data/' + '*.h5')
    list_links = [l.strip() for l in f]
    for url in list_links:
        link = url
        dict_file = extract_one_recipe.get_recipe(link)
        print_json(link, dict_file)

        dict_file['Recipe'] = from_list_to_str(dict_file['Recipe'])

        # transform to data set (funny tiny dataset;)
        df = load_data_transform_to_set(dict_file)
        text = df['paragraph']

        one_page_data_path = preprocess_clean_data(df, 'one_page')
        one_page_set_clean  = pd.read_pickle(one_page_data_path)

        # sent to sequence only for  NLP set
        sent2vec_one_page= sent2vec(one_page_set_clean.remove_stop_words, MAX_SEQ_LEN, VOCAB_SIZE)
        # for other features set
        X_meta_one_page = one_page_set_clean[['sent_count', 'num_count', 'clean_paragraph_len', 'verb_count','contains_pron']]
        y_one_page = one_page_set_clean['label']
        for model in models_list:
            print(link)
            eval_on_one_page(sent2vec_one_page, X_meta_one_page, y_one_page, model,text)











