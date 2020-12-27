import datetime
import logging
import os
import pickle
from config import DATA_FILE, LOG_FILE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def timeit(method):
    """The function of time measure
    :param method:
    :return print into logfile and into console ide
    __name__ of the function and execution time"""

    def timed(*args, **kw):
        begin = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        result = method(*args, **kw)
        end = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            total_time = datetime.datetime.strptime(end, '%H:%M:%S.%f') - datetime.datetime.strptime(begin, '%H:%M:%S.%f')
            kw['log_time'][name] = total_time
        else:
            total_time = datetime.datetime.strptime(end, '%H:%M:%S.%f') - datetime.datetime.strptime(begin, '%H:%M:%S.%f')
            logging.info(f"Total time taken in : {method.__name__} {total_time}")
            print('Total time taken in : {}  {} ms'.format(method.__name__, total_time))
        return result

    return timed

@timeit
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

@timeit
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


@timeit
def save_data_to_pkl(data_file, file_name=DATA_FILE):
    """The function receive  a data_file:obj
    checks the path and dumps into the pkl file
    :return full path of saved data_file:str"""

    file, path = check_dir_path(file_name, 'wb')
    pickle.dump(data_file, file, pickle.HIGHEST_PROTOCOL)
    file.close()

    return path


@timeit
def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
