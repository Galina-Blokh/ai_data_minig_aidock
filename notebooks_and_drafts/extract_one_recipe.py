import logging

import grequests
from bs4 import BeautifulSoup
from config import BATCHES, EMPTY_LINKS, LOG_FILE
from notebooks_and_drafts.extract_all_links import extract_links_to_file

# log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# @profile
def get_recipe(url_to_get, counter_to_print=1):
    """  To collect ingredients and  INSTRUCTIONS text  from a single recipe page
    :param   url_to_get: str,
    :param   counter_to_print: int default 1  - for one_link_run.py
    :return  json_file: dict(str:list[str] ,str:str) /'Recipe':list[str], 'INGREDIENTS': str'
    """
    response = ''
    json_file = {}  # not real json because of utf-8 encoding
    ingredients_list = []
    instructions_list = []
    try:
        page = grequests.get(url_to_get)
        response = grequests.map([page], size=BATCHES)
    except:
        logging.info(f"Can't collect `recipe` from a page {url_to_get}")
    for res in response:
        soup = BeautifulSoup(res.content, 'html.parser')
        # get Recipe ingredients from recipe page into list
        recipe_ingredients_lines = soup.find_all(attrs={'class': ['wprm-recipe-ingredient',
                                                                  'ingredient']})
        # get INSTRUCTIONS from recipe page
        recipe_instructions_lines = soup.find_all(attrs={'class': ['wprm-recipe-instruction-text',
                                                                   'instruction']})
        # add broken link into txt file and log, remove from all_links.txt
        if len(recipe_ingredients_lines) < 1 or len(recipe_instructions_lines) < 1:
            logging.info(f'No ingredients or INSTRUCTIONS on recipe page {counter_to_print}. {url_to_get}')
            extract_links_to_file(EMPTY_LINKS, url_to_get)
        for item, paragraph in zip(recipe_ingredients_lines, recipe_instructions_lines):
            # add ingredients into list
            ingredients_list.append(item.text)
            # add INSTRUCTIONS into list
            instructions_list.append(paragraph.text.replace('\n', '\n\n'))
        # add ingredients_list into dict
        json_file['Recipe'] = ingredients_list
        # transform INSTRUCTIONS into string and add into dict
        json_file['INSTRUCTIONS'] = '\n\n'.join(instructions_list)
    logging.info(f'Recipe from recipe page {counter_to_print}. {url_to_get} Collected')
    return json_file


if __name__ == '__main__':
    pass
