from collections import defaultdict
import grequests
from bs4 import BeautifulSoup


def get_one(url_to_get):
    """  To collect ingredients and  INSTRUCTIONS text  from a single recipe page
    :param   url_to_get: str,
    :return  json_file: defaultdict('Recipe':list[list[str]], 'INGREDIENTS': list[str]')
    """
    json_file = defaultdict(list)
    page = [grequests.get(url_to_get)]

    # get data from website
    response = grequests.map(page, size=20)
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

        # collect data into dictionary
        json_file['Recipe'].append(ingredients_list)
        json_file['INSTRUCTIONS'].append('\n\n'.join(instructions_list))
    return json_file
