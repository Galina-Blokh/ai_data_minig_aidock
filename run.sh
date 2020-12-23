#!/bin/bash
BATCHES=20
LINK_PATTERN=https://www.loveandlemons
URL=https://www.loveandlemons.com/recipes/
LOG_FILE=recipes_logging.log
FILE_LINKS_NAME=all_recipes_links.txt
EMPTY_LINKS=no_recipe_page.txt
DATA_FILE=recipes.pkl
LINK=$1
python for_one_link_run.py "${LINK}"
