import datetime
import sys

import argparse
from extract_one_recipe import *

# # log-file will be created in the same dir
logging.basicConfig(filename=LOG_FILE, level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():


    json_file = []
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    path_to_all_links = extract_links_to_file()
    stop_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    total_time = (datetime.datetime.strptime(stop_time, '%H:%M:%S.%f') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S.%f'))
    logging.info(f"Collecting all links was executed {total_time} ")

    start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    recipe_links = open(path_to_all_links, "r").readlines()
    # json_file = get_recipe('https://www.loveandlemons.com/watermelon-gazpacho/', 1)
    # print_json('https://www.loveandlemons.com/watermelon-gazpacho/', json_file)
    for counter_to_print, link in enumerate(recipe_links):
        one_recipe = get_recipe(link, counter_to_print)  # TODO remove later  print json
        json_file.append(one_recipe)
        print_json(link, one_recipe)

    get_recipes = save_data_to_pkl(json_file)

    logging.info(f'All data from recipes pages was saved into {get_recipes}')
    stop_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    total_time = (datetime.datetime.strptime(stop_time, '%H:%M:%S.%f') -
                  datetime.datetime.strptime(start_time,'%H:%M:%S.%f'))
    logging.info(f"Collecting was executed {total_time} seconds")
    sys.exit('This is the end...')


if __name__ == '__main__':
    main()
