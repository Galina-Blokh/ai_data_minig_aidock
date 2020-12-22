import grequests
from bs4 import BeautifulSoup
import config
from extract_one_recipe import check_dir_path

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

    soup = [BeautifulSoup(res.text, "html.parser") for res in response]
    recipes_links = [link.get("href") for link in soup[0].find_all('a') if
                     str(link.get("href")).startswith(config.LINK_PATTERN)]

    logger.info(f"Collected {len(recipes_links)} `recipes_links` from recipes page")
    return recipes_links


def extract_links_to_file(file_name=config.FILE_LINKS_NAME):
    """
    Function receives a file_name:str
    extract links to file_name.txt file
    :return: void
    """
    output_links, path = check_dir_path(file_name, 'w+')
    all_links = get_all_links_recipes(config.URL)

    # writing down all links into txt file
    [output_links.write(link + '\n') for link in all_links]
    output_links.close()
    logger.info(f"Links were written into file {path} finished")
    return path

if __name__ == '__main__':
    pass
