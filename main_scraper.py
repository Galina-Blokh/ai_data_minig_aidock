import sys
from extract_one_recipe import *
from utils import save_data_to_pkl, timeit

#  log-file will be created in the same dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@timeit
def main():
    json_file = []
    # extracting all links

    path_to_all_links = extract_links_to_file()
    logging.info(f"Collecting all links was executed ")

    # writing down recipe data from each page
    recipe_links = open(path_to_all_links, "r").readlines()
    for counter_to_print, link in enumerate(recipe_links):
        one_recipe = get_recipe(link, counter_to_print)
        json_file.append(one_recipe)

    # save the pkl file
    get_recipes = save_data_to_pkl(json_file)
    logging.info(f'All data from recipes pages was saved into {get_recipes}')

    logging.info(f"Collecting was executed ")
    sys.exit('This is the end...')


if __name__ == '__main__':
    main()
