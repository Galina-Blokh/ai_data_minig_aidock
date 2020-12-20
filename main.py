import grequests
from bs4 import BeautifulSoup
import config
import time

logger = config.get_logger(__name__)


def get_all_links_recipes(url_to_get):
    """The function receives url_to_get:str
     collects all recipes links from page with all recipes
     :returns recipes_links:list[str] """
    try:
        page = grequests.get(url_to_get)
        response = grequests.map([page], size=config.BATCHES)
    except:
        logger.warning("Can't collect `recipes_links` from a page")

    soup = [BeautifulSoup(res.text, 'html.parser') for res in response]
    recipes_links = [link.get('href') for link in soup[0].find_all('a') if
                     str(link.get('href')).startswith(config.LINK_PATTERN)]

    logger.info(f"Collected {len(recipes_links)} `recipes_links` from recipes page")
    return recipes_links


def extract_links_to_file(file_name=config.FILE_LINKS_NAME):
    """
    Function receives a file_name:str
    extract links to file_name.txt file
    :return: void
    """
    all_links = get_all_links_recipes(config.URL)
    output_links = open(file_name, 'w')
    for link in all_links:
        output_links.write(link + '\n')
    output_links.close()
    logger.info(f'Links were written into file {config.FILE_LINKS_NAME} finished')


if __name__ == '__main__':
    start_time = time.process_time()
    extract_links_to_file()
    logger.info(f"The program was executed {time.process_time() - start_time} seconds")

