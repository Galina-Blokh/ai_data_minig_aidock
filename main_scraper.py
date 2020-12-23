import datetime
import sys
from extract_all_links import extract_links_to_file
from extract_one_recipe import *

# # log-file will be created in the same dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    json_file = []
    # extracting all links
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    path_to_all_links = extract_links_to_file()
    stop_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    total_time = (datetime.datetime.strptime(stop_time, '%H:%M:%S.%f') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S.%f'))
    logging.info(f"Collecting all links was executed {total_time} ")

    # writing down recipe data from each page
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    recipe_links = open(path_to_all_links, "r").readlines()
    for counter_to_print, link in enumerate(recipe_links):
        one_recipe = get_recipe(link, counter_to_print)
        json_file.append(one_recipe)

    # save the pkl file
    get_recipes = save_data_to_pkl(json_file)
    logging.info(f'All data from recipes pages was saved into {get_recipes}')
    stop_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
    total_time = (datetime.datetime.strptime(stop_time, '%H:%M:%S.%f') -
                  datetime.datetime.strptime(start_time, '%H:%M:%S.%f'))
    logging.info(f"Collecting was executed {total_time}")
    sys.exit('This is the end...')


if __name__ == '__main__':
    main()
