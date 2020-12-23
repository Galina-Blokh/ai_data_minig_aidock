import logging
import os
import pickle
import grequests
from bs4 import BeautifulSoup
from config import DATA_FILE, BATCHES, EMPTY_LINKS, LOG_FILE

# # log-file will be created in the same dir
from extract_all_links import extract_links_to_file

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def print_json(url_to_get_recipe, json_file):
    """The function receive url_to_get_recipe:str and json_file:dict
    prints the pretty json file format
    :return void"""

    print(f'Url: {url_to_get_recipe}' + '{\t\t' + list(json_file.keys())[0] + ':' + '\n\t\t\t\t[')
    for k in json_file['Recipe']:
        print('\t\t\t\t\t\t' + k)

    print('\t\t\t\t]')
    print('\n\t' + list(json_file.keys())[1] + ':')
    print('"' + json_file['INSTRUCTIONS'] + '"')
    print('}\n')


def check_dir_path(filename, what_to_do):
    """The function receive filename:txt
    checks is the path exists and creates empty file
    :return file and path"""
    path = ''
    if not os.path.exists(f'{os.getcwd()}/data'):
        os.makedirs(f'{os.getcwd()}/data')
    try:
        path = f'{os.getcwd()}/data/{filename}'
        file = open(path, what_to_do)
    except:
        raise Exception(f'WTF!!!! Could not open {path}.')
    return file, path  # DON'T FORGET TO CLOSE `file` IN THE PLACE WHERE YOU CALL THIS FUNCTION


def save_data_to_pkl(json_file):
    """The function receive  a json_file: Dict[str, Union[list, str]] = {}
    checks the path and dumps into the pkl file
    :return full path of saved file"""

    file, path = check_dir_path(DATA_FILE, 'wb')
    pickle.dump(json_file, file)
    file.close()

    return path


def get_recipe(url_to_get, counter_to_print=1):
    """The function receives url_to_get:str to the one recipe page
    and counter_to_print:int in log
     collects ingredients and  INSTRUCTIONS text  from a recipe page
     :returns  a dict() with recipe_ingredients:list[str] and instructions:str """
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
        recipe_ingredients_lines = soup.find_all(attrs={'class': ['wprm-recipe-ingredient', 'ingredient']})

        # get INSTRUCTIONS from recipe page
        recipe_instructions_lines = soup.find_all(attrs={'class': ['wprm-recipe-instruction-text', 'instruction']})
        # add broken link into txt file and log
        if len(recipe_ingredients_lines) < 1 or len(recipe_instructions_lines) < 1:
            logging.info(f'No ingredients or INSTRUCTIONS on recipe page {counter_to_print}. {url_to_get}')
            extract_links_to_file(EMPTY_LINKS, url_to_get)

        for item, paragraph in zip(recipe_ingredients_lines, recipe_instructions_lines):
            # add ingredients into list
            ingredients_list.append(item.text)
            # INSTRUCTIONS into list
            instructions_list.append(paragraph.text.replace('\n', '\n\n'))

        # add ingredients_list into dict
        json_file['Recipe'] = ingredients_list
        # transform INSTRUCTIONS into string and add into dict
        json_file['INSTRUCTIONS'] = '\n\n'.join(instructions_list)

    logging.info(f'Recipe from recipe page {counter_to_print}.{url_to_get} Collected')
    print_json(url_to_get, json_file)  # FOR BASH SCRIPT RUN
    return json_file


if __name__ == '__main__':
    pass
