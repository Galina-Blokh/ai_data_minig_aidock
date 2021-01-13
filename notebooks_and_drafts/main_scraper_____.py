import sys
import time

from notebooks_and_drafts.extract_one_recipe import *
from utils import save_data_to_pkl, profile, elapsed_since

#  log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@profile
def main_scrapper():
    """
    To extract all links from a page with all recipes,
    write down collected links into data/all_links.txt file
    redact the all_links.txt file --> leave only links with no duplicates and pages  with data.
    then read links from file and collect data from each recipe page
    save collected data into pkl file
    If you want to continue run the project, then run `preprocess.py`
    """
    start = time.time()

    logging.info(f"Starting collecting all links...")

    json_file = []

    # extracting all links
    path_to_all_links = extract_links_to_file()
    elapsed_time1 = elapsed_since(start)
    logging.info(f"Collecting all links was executed {elapsed_time1}")

    # writing down recipe data from each page
    # input('This stop run just for taking out several links from all_links.txt to test at the end. if you made it press enter') #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    recipe_links = open(path_to_all_links, "r").readlines()
    for counter_to_print, link in enumerate(recipe_links):
        one_recipe = get_recipe(link, counter_to_print)
        json_file.append(one_recipe)

    # save the pkl file
    get_recipes = save_data_to_pkl(json_file)
    logging.info(f'All data from recipes pages was saved into {get_recipes}')
    elapsed_time2 = elapsed_since(start)
    logging.info(f"Collecting was executed {elapsed_time2}\nIf you want to continue, run `model_preprocess/preprocess.py`")
    sys.exit('This is the end of scraping...\nIf you want to continue, run `model_preprocess/preprocess.py`')


if __name__ == '__main__':
    main_scrapper()
