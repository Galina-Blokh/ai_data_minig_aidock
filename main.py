import time

from config import FILE_LINKS_NAME ,get_logger
from extract_all_links import *
from extract_one_recipe import *

logger = get_logger(__name__)


def main():
    json_file = []
    start_time = time.process_time()
    path_to_all_links = extract_links_to_file()
    logger.info(f"Collecting all links was executed {time.process_time() - start_time} seconds")

    start_time = time.process_time()
    recipe_links = open(path_to_all_links, "r").readlines()

    for counter_to_print, link in enumerate(recipe_links):
        json_file = get_recipe(link,counter_to_print)
        print_json(link, json_file)

    get_recipes = save_data_to_pkl(json_file)

    logger.info(f"All data from recipes pages was saved into {get_recipes} ")
    logger.info(f"Collecting was executed {time.process_time() - start_time} seconds")


if __name__ == '__main__':
    main()
