import grequests
from bs4 import BeautifulSoup
from config import *

logger = get_logger(__name__)


def find_all_links_recipes(url_to_get):
    """The function receives url_to_get:str
     collects all recipes links from page with all recipes
     :returns recipes_links:list[str] """
    try:
        page = grequests.get(url_to_get)
        response = grequests.map([page], size=BATCHES)
    except:
        logger.info("Can't collect `recipes_links` from a page")
        teams = '0'
    soup = [BeautifulSoup(res.text, 'html.parser') for res in response]
    recipes_links = [link.get('href') for link in soup[0].find_all('a') if
                     str(link.get('href')).startswith(LINK_PATTERN)]

    logger.info(f"Collected {len(recipes_links)} `recipes_links` from recipes page")
    return recipes_links


if __name__ == '__main__':
    find_all_links_recipes(URL)
