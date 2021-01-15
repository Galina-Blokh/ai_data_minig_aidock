import logging
import sys
from collections import defaultdict
import grequests
from bs4 import BeautifulSoup
import config
from utils import save_data_to_pkl, profile

#  log-file will be created in the main dir
logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@profile
def get_all_links_recipes(url_to_get):
    """To collect all recipes links from web site
     :param url_to_get: str
     :return recipes_links: set[str]
     """
    try:
        page = grequests.get(url_to_get)
        response = grequests.map([page], size=config.BATCHES)
    except AttributeError:
        logging.info("Can't collect `recipes_links` from a page")
    soup = [BeautifulSoup(res.text, 'html.parser') for res in response]
    recipes_links = [link.get('href') for link in soup[0].find_all('a') if
                     str(link.get('href')).startswith(config.LINK_PATTERN)]
    logging.info(f'Collected {len(set(recipes_links))} `recipes_links` from recipes page')
    return set(recipes_links)


@profile
def get_all_recipes(url_to_get=config.URL):
    """
      To extract all links from a page with all recipes,
      write down 10 first urls into data/test_links.txt file
      collect data from each recipe page
      save collected data into pkl file
      If you want to continue the project run `preprocess.py`
      """
    json_file = defaultdict(list)

    logging.info(f"Starting collecting all links...")
    # get all urls
    urls = list(get_all_links_recipes(url_to_get))
    # take out 10 url for model testing
    with open('data/' + config.TEST_LINKS_FILE, "w") as f:
        [f.write(l + '\n') for l in urls[:10]]
    links = urls[10:]
    logging.info(f'Took out {len(urls)-len(links)} recipes_links for testing, left {len(links)} links')
    page = (grequests.get(u) for u in links)
    # get data from website
    response = grequests.map(page, size=config.BATCHES)
    for res in response:
        try:
            soup = BeautifulSoup(res.content, 'html.parser')
        except AttributeError:
            continue
        recipe_ingredients_lines = soup.find_all(attrs={'class': ['wprm-recipe-ingredient',
                                                                  'ingredient']})
        recipe_instructions_lines = soup.find_all(attrs={'class': ['wprm-recipe-instruction-text',
                                                                   'instruction']})
        ingredients_list = [item.text for item in recipe_ingredients_lines]
        instructions_list = [paragraph.text for paragraph in recipe_instructions_lines]

        # collect data into dictionary and save into pkl
        json_file['Recipe'].append(ingredients_list)
        json_file['INSTRUCTIONS'].append('\n\n'.join(instructions_list))
    path_to_data = save_data_to_pkl(json_file, 'new_recipe.pkl')

    return json_file,path_to_data


if __name__ == '__main__':
    _,path = get_all_recipes()
    logging.info(f'All data from recipes pages was saved into \n{path}\nIf you want to continue, run '
                 f'`preprocess.py`')
    sys.exit('This is the end of scraping...\nIf you want to continue, run `preprocess.py`')
