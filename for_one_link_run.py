import argparse
import os
import pandas as pd
from scrapping import extract_one_recipe
from config import MAX_SEQ_LEN, VOCAB_SIZE, MODEL_NAME
from model_preprocess.preprocess import preprocess_clean_data, load_data_transform_to_set, from_list_to_str, sent2vec
from model_preprocess.run_tensorflow import eval_on_one_page
from scrapping.extract_one_recipe import get_recipe
from utils import print_json


# @profile
def main_for_one_link():
    parser = argparse.ArgumentParser(description='Print the recipe json from given link')
    parser.add_argument('link')
    args = parser.parse_args()
    # link = 'https://www.loveandlemons.com/po-boy-sandwich/'
    dict_file = get_recipe(str(args.link).strip())
    # dict_file = extract_one_recipe.get_recipe(link)
    # print_json(link, dict_file)

    dict_file['Recipe'] = from_list_to_str(dict_file['Recipe'])

    # transform to data set (funny dataset)
    df = load_data_transform_to_set(dict_file)
    text = df['paragraph']
    one_page_data_path = preprocess_clean_data(df, 'one_page')
    one_page_set_clean = pd.read_pickle(one_page_data_path)

    # sent to sequence only for  NLP TRAIN
    sent2vec_one_page = sent2vec(one_page_set_clean.remove_stop_words, MAX_SEQ_LEN, VOCAB_SIZE)
    # for other features train
    X_meta_one_page = one_page_set_clean[['sent_count', 'num_count', 'clean_paragraph_len', 'contains_pron']]
    y_one_page = one_page_set_clean['label']


    model = f'{os.getcwd()}{MODEL_NAME}'
    eval_on_one_page(sent2vec_one_page, X_meta_one_page, y_one_page, model,text,one_page_set_clean.remove_stop_words)

    print_json(str(args.link).strip(), dict_file)
#     # DONEclean-preprocess data from link
#     # DONEload pre-trained model
#     #  DONE predict on new set
#



if __name__ == '__main__':

    main_for_one_link()

    # DONE add profile
    # TODO add logging
#     # TODO print to file/console info about prediction