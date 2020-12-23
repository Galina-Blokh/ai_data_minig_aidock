import logging
BATCHES = 20
LINK_PATTERN = 'https://www.loveandlemons'
URL = 'https://www.loveandlemons.com/recipes/'
LOG_FILE = 'recipes_logging.log'
FILE_LINKS_NAME = 'all_recipes_links.txt'
EMPTY_LINKS = 'no_recipe_page.txt'
DATA_FILE = 'recipes.pkl'

# # log-file will be created in the same dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# TODO add requirements
