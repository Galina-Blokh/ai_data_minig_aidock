import datetime
import sys
import time

from extract_one_recipe import *
from utils import save_data_to_pkl, profile, elapsed_since

#  log-file will be created in the same dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@profile
def main_scrapper():
    start = time.time()

    logging.info(f"Starting collecting all links...")
    print(f"Starting collecting all links...")

    json_file = []

    # extracting all links
    path_to_all_links = extract_links_to_file()
    elapsed_time1 = elapsed_since(start)
    logging.info(f"Collecting all links was executed {elapsed_time1}")
    print(f"Collecting all links was executed {elapsed_time1}")

    # writing down recipe data from each page
    recipe_links = open(path_to_all_links, "r").readlines()
    for counter_to_print, link in enumerate(recipe_links):
        one_recipe = get_recipe(link, counter_to_print)
        json_file.append(one_recipe)

    # save the pkl file
    get_recipes = save_data_to_pkl(json_file)
    logging.info(f'All data from recipes pages was saved into {get_recipes}')
    elapsed_time2 = elapsed_since(start)
    print(f"Collecting was executed {elapsed_time2}")
    logging.info(f"Collecting was executed {elapsed_time2}")
    sys.exit('This is the end...')


if __name__ == '__main__':
    main_scrapper()
