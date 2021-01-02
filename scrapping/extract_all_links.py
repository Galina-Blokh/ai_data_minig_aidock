import logging
import os
import grequests
from bs4 import BeautifulSoup
import config
from utils import check_dir_path, profile

# # log-file will be created in the main dir
logging.basicConfig(filename=os.pardir+config.LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@profile
def get_all_links_recipes(url_to_get):
    """To collect all recipes links from page with all recipes
     :param url_to_get: str
     :return recipes_links: list[str]
     """
    response = ''
    try:
        page = grequests.get(url_to_get)
        response = grequests.map([page], size=config.BATCHES)
    except:
        logging.info("Can't collect `recipes_links` from a page")

    soup = [BeautifulSoup(res.text, 'html.parser') for res in response]
    recipes_links = [link.get('href') for link in soup[0].find_all('a') if
                     str(link.get('href')).startswith(config.LINK_PATTERN)]

    logging.info(f'Collected {len(set(recipes_links))} `recipes_links` from recipes page')
    return set(recipes_links)


@profile
def extract_links_to_file(file_name=config.FILE_LINKS_NAME, url_to_write=config.URL):
    """
    To write down links   to all_links.txt or empty_links.txt file
    if link is in empty_links file, then this link removed from all_links.txt
    :param file_name: str / default config.FILE_LINKS_NAME,
    :param  url_to_write: str / default config.URL
    :return path:str where link was written
    """
    output_links, path = check_dir_path(file_name, 'a+')

    if file_name == config.FILE_LINKS_NAME:
        all_links = get_all_links_recipes(url_to_write)
        # writing down all links into txt file
        [output_links.write(link + '\n') for link in all_links]

    elif file_name == config.EMPTY_LINKS:
        output_links.write(url_to_write)

        # deleting empty links from all_recipe_links.txt
        all_links_path = f'{os.pardir}/data/{config.FILE_LINKS_NAME}'
        with open(all_links_path, "r+") as f:
            new_f = f.readlines()
            f.seek(0)
            for line in new_f:
                if url_to_write not in line:
                    f.write(line)
            f.truncate()

    output_links.close()
    logging.info(f'Links were written into file {path} finished')
    return path


if __name__ == '__main__':
    pass
