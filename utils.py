import logging
import pickle
import time
import os
import psutil
import inspect
from config import DATA_FILE, LOG_FILE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def elapsed_since(start):
    """time measurement
    :param start: str - time of start
    :return diff execution time: str"""
    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed * 1000, 2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed / 60, 2)) + "min"
    else:
        return str(round(elapsed / 3600, 2)) + "hrs"


def get_process_memory():
    """Function of memory information about the process.
        :return The "portable" fields( available on all platforms) `rss` and `vms`.
        All numbers are expressed in bytes."""

    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms


def format_bytes(bytes):
    """Function to transform bytes into B,kB,Mb,Gb
    :param bytes:int
    :return str with transformed bytes """

    if abs(bytes) < 1000:
        return str(bytes) + "B"
    elif abs(bytes) < 1e6:
        return str(round(bytes / 1e3, 2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def profile(func, *args, **kwargs):
    """The function of time measure
    :param func:
    :return print into logfile and into console ide
    __name__ of the function and execution time amd memory usage
     source link of this function:
     https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python"""

    def wrapper(*args, **kwargs):
        rss_before, vms_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after = get_process_memory()
        print("Profiling: {:>20}  RSS: {:>8} | VMS: {:>8} | time: {:>8}"
              .format("<" + func.__name__ + ">",
                      format_bytes(rss_after - rss_before),
                      format_bytes(vms_after - vms_before),
                      elapsed_time))
        logging.info("Profiling: {:>20}  RSS: {:>8} | VMS: {:>8} | time: {:>8}"
                      .format("<" + func.__name__ + ">",
                              format_bytes(rss_after - rss_before),
                              format_bytes(vms_after - vms_before),
                              elapsed_time))
        return result

    if inspect.isfunction(func):
        return wrapper
    elif inspect.ismethod(func):
        return wrapper(*args, **kwargs)


@profile
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


@profile
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


@profile
def save_data_to_pkl(data_file, file_name=DATA_FILE):
    """The function receive  a data_file:obj
    checks the path and dumps into the pkl file
    :return full path of saved data_file:str"""

    file, path = check_dir_path(file_name, 'wb')
    pickle.dump(data_file, file, pickle.HIGHEST_PROTOCOL)
    file.close()
    return path


@profile
def read_from_pickle(filename):
    """The function receive  a data_file_name:str
    read from the pkl file
    :return pkl.load object"""

    with open(filename, 'rb') as f:
        return pickle.load(f)
