import logging
import os
import pickle

from config import DATA_FILE, LOG_FILE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def print_json(url_to_get_recipe, json_file):
    """The function receive url_to_get_recipe:str and json_file:dict
    prints the pretty json file format
    :return void"""

    print(u"Url: {} \n{{\n \t\t{} :\n\t\t\t\t[".format(url_to_get_recipe, list(json_file.keys())[0]))
    for k in json_file['Recipe']:
        print(u'\t\t\t\t\t\t {}'.format(str(k).strip()))
    print(u'\t\t\t\t]')
    print('\n\t\t' + str(list(json_file.keys())[1] + ':'))
    print('"' + json_file['INSTRUCTIONS'] + '"')
    print('}\n')


def check_dir_path(filename, what_to_do):
    """The function receive filename:txt
    checks is the path exists and creates empty file
    :return file and path"""
    path = ''
    if not os.path.exists('{}/data'.format(os.getcwd())):
        os.makedirs('{}/data'.format(os.getcwd()))
    try:
        path = '{}/data/{}'.format(os.getcwd(), filename)
        file = open(path, what_to_do)
    except:
        raise Exception('WTF!!!! Could not open {}'.format(path))
    return file, path  # DON'T FORGET TO CLOSE `file` IN THE PLACE WHERE YOU CALL THIS FUNCTION


def save_data_to_pkl(json_file):
    """The function receive  a json_file: Dict[str, Union[list, str]] = {}
    checks the path and dumps into the pkl file
    :return full path of saved file"""

    file, path = check_dir_path(DATA_FILE, 'wb')
    pickle.dump(json_file, file, pickle.HIGHEST_PROTOCOL)
    file.close()

    return path


def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
